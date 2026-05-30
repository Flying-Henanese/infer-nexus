# Qwen3 Reranker, vLLM, And Gateway Notes

## Summary

This note captures the useful findings from the gateway and model validation work, especially:

- how Qwen3 chat models interact with vLLM reasoning controls
- why Qwen3-Reranker is harder to adapt than ordinary rerankers
- which vLLM parameters matter for rerank/scoring models
- what a reranker actually does in a retrieval pipeline
- what we learned from comparing `Qwen3-Reranker-8B` and `bge-reranker`

Relevant references:

- Qwen3-Reranker-8B model card: <https://huggingface.co/Qwen/Qwen3-Reranker-8B>
- vLLM scoring docs: <https://docs.vllm.ai/en/stable/models/pooling_models/scoring/>
- vLLM reasoning docs: <https://docs.vllm.ai/en/latest/features/reasoning_outputs/>

## 1. High-level conclusions

The most important conclusions from this round:

1. `Qwen3-32B` style chat models are relatively straightforward on vLLM. The main adaptation point is reasoning/thinking control.
2. `Qwen3-Reranker-8B` is not a conventional reranker architecture, so vLLM integration is significantly more fragile than with ordinary rerankers.
3. For `Qwen3-Reranker`, "model starts" and "API returns scores" do not imply "ranking is correct".
4. `bge-reranker` behaved much more reliably in the current environment, which strongly suggests the main problem was not the gateway itself, but the `Qwen3-Reranker + vLLM` combination.

## 2. What a reranker does

A reranker is not the first-stage retriever. It is the second-stage ranker.

Typical retrieval pipeline:

1. Use `embedding` to vectorize the query and documents.
2. Run ANN / vector retrieval to get Top-K candidates quickly.
3. Feed `(query, document)` pairs into a `reranker`.
4. The reranker outputs a relevance score for each pair and reorders the candidates.

So:

- `embedding` is optimized for recall and speed
- `reranker` is optimized for precision

There are two common reranker families:

1. Traditional cross-encoder / sequence-classification rerankers
   Input is `(query, doc)` and output is a score.
2. Generative-LLM-based rerankers
   The base model is still a causal LM, but scoring is derived from prompt formatting and token logits.

`Qwen3-Reranker` belongs to the second family. This is the root reason it is harder to serve correctly than many BGE rerankers.

## 3. Why Qwen3-Reranker is awkward to serve

The model card itself exposes the key mismatch:

- its Hugging Face tags include `text-generation`
- the official Transformers usage uses `AutoModelForCausalLM`
- but the task is `Text Reranking`

In other words, it is not "natively shaped" like a standard sequence classification reranker. It is built on top of a Qwen3 generative backbone and adapted into a ranking model.

The official model card recommends:

- `sentence-transformers.CrossEncoder`
- or the provided Transformers path

Model card references:

- `CrossEncoder("Qwen/Qwen3-Reranker-8B")`
- default prompt injection for `"query"`

Source:

- <https://huggingface.co/Qwen/Qwen3-Reranker-8B>

Important implication:

- prompt/template formatting matters a lot
- serving logic must understand the scoring mechanism
- blindly treating it like a generic `/score` model is risky

## 4. Why vLLM needs special flags for Qwen3-Reranker

For `Qwen3-Reranker`, vLLM documents explicitly require these core overrides:

```json
{
  "architectures": ["Qwen3ForSequenceClassification"],
  "classifier_from_token": ["no", "yes"],
  "is_original_qwen3_reranker": true
}
```

And typically also:

- `--runner pooling`
- `--chat-template examples/pooling/score/template/qwen3_reranker.jinja`

Relevant docs:

- latest scoring docs: <https://docs.vllm.ai/en/stable/models/pooling_models/scoring/>
- historical examples: <https://docs.vllm.ai/en/v0.19.1/examples/pooling/score/>

What each parameter means:

- `architectures: ["Qwen3ForSequenceClassification"]`
  Force vLLM to interpret the model as sequence classification rather than the default causal LM architecture.
- `classifier_from_token: ["no", "yes"]`
  Tell vLLM to derive classification from the relevant token representations/logits associated with `"no"` and `"yes"`.
- `is_original_qwen3_reranker: true`
  Enable the special conversion logic vLLM uses for the original Qwen3 reranker implementation.
- `runner="pooling"`
  In many vLLM versions, score/rerank models are expected to run through the pooling runner rather than the ordinary generation runner.
- `chat_template=qwen3_reranker.jinja`
  Format each `(query, doc)` pair into the exact prompt structure expected by the model.

Official template source:

- <https://raw.githubusercontent.com/vllm-project/vllm/main/examples/pooling/score/template/qwen3_reranker.jinja>

## 5. Why chat_template alone was not enough

During debugging we verified that:

- `chat_template` was indeed passed into `LLM(...)`
- but raw `llm.score(...)` still produced obviously bad rankings

This means the problem was not only "template path not loaded".

In some vLLM versions, two separate things matter:

1. passing the template at model initialization time
2. explicitly passing the template again at `score(...)` call time

This is an important engineering lesson:

- "startup parameters look correct" does not necessarily mean the scoring path is actually using them the way you expect

## 6. Qwen3 chat models and reasoning/thinking controls

This part worked correctly in the current project.

To suppress `<think>...</think>` output for Qwen3 chat models in vLLM, the key control is:

```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }
}
```

Source:

- vLLM reasoning docs: <https://docs.vllm.ai/en/latest/features/reasoning_outputs/>

Practical notes:

- prompt-level switches like `/no_think` are not reliable enough
- template-level control is the stable mechanism
- if a gateway sits in front of vLLM, it must allow reasoning-related fields to pass through

That is why the gateway changes around:

- `request_policy.allow_reasoning`
- `request_defaults.chat_template_kwargs.enable_thinking=false`

were necessary.

## 7. Key vLLM adaptation checklist

### Chat models

- Standard chat models typically work with ordinary `LLM(...)` generation setup.
- For Qwen3 thinking suppression, use:

```json
extra_body.chat_template_kwargs.enable_thinking = false
```

### Embedding models

- Use the embedding endpoint, not chat.
- `completion_tokens=0` is expected because no text is generated.
- Output dimension is model-specific, for example `Qwen3-Embedding-8B` returned dimension `4096`.

### Rerank models

- Not every model whose name includes "reranker" will be correctly interpreted by vLLM by default.
- Some models require:
  - `runner="pooling"`
  - `chat_template`
  - `hf_overrides`

### Paths

- `chat_template` should preferably be an absolute path.
- Relative paths are fragile because they depend on the process working directory.

### Version differences

- Newer vLLM docs often emphasize `task="score"`.
- Older versions and many practical examples still rely on `runner="pooling"`.
- You cannot safely copy the latest API style without checking the actual deployed `vllm.__version__`.

## 8. Why the Qwen3-Reranker issue was not a gateway bug

The most decisive isolation step was:

- not only testing `/v1/rerank`
- but also bypassing the gateway and calling raw `vllm.LLM(...).score(...)`

If raw `score()` already ranks:

- `Paris is the capital of France.`

below irrelevant text, then:

- the gateway is not "corrupting" a good model result
- the underlying `Qwen3-Reranker + vLLM` behavior itself is already unreliable

This isolation pattern is useful for future debugging too:

1. test the OpenAI-compatible API
2. test the gateway/backend integration layer
3. test the raw inference library directly

## 9. Why BGE reranker behaved better

`bge-reranker` is much closer to the traditional reranker/cross-encoder pattern and appears to have fewer serving-path mismatches in vLLM.

In this environment:

- `bge-reranker` consistently ranked the correct passage first
- `Qwen3-Reranker-8B` produced obviously incorrect rankings under the same gateway

This suggests a practical engineering rule:

- if the goal is to get reranking stable first, use a more mature cross-encoder-style reranker first
- treat Qwen3-Reranker as a special compatibility project, not the baseline

## 10. Public signals that Qwen3-Reranker on vLLM is sensitive

The debugging results matched public reports.

Examples:

- users reported incorrect scores and poor/random rankings for `Qwen3-Reranker` in vLLM
  - issue #21681
  - <https://github.com/vllm-project/vllm/issues/21681>
- users explicitly stated that the `hf_overrides`, `runner`, and `chat-template` flags are mandatory for Qwen3-Reranker in vLLM
  - issue #33970
  - <https://github.com/vllm-project/vllm/issues/33970>
- similar wrong-score behavior has also been reported for Qwen3-VL reranker under vLLM
  - issue #35412
  - <https://github.com/vllm-project/vllm/issues/35412>

So the right mental model is:

- `Qwen3-Reranker` is not "unsupported"
- but it is "special-case supported, compatibility-sensitive, and correctness must be validated independently"

## 11. Practical recommendations

### For Qwen3 chat

- keep `chat_template_kwargs.enable_thinking=false`
- let the gateway pass reasoning-related request fields when needed

### For Qwen3-Reranker

- do not use "it starts successfully" as the success criterion
- always run fixed query/document correctness tests
- keep a Transformers or sentence-transformers comparison path available

### For rerank production readiness

- use something like `bge-reranker` first to validate the gateway path
- treat `Qwen3-Reranker` as an optimization or later compatibility target

### For vLLM version management

- document the exact runner/template/hf_overrides requirements for each special model
- do not assume upgrading vLLM automatically preserves behavior

## 12. Notes about the current project context

From this project round specifically:

- Qwen3 chat reasoning control was successfully integrated and validated.
- Local vLLM chat streaming support was added in the gateway path.
- Error mapping and error-message cleaning were improved for chat, embedding, and rerank request validation failures.
- `Qwen3-Reranker-8B` remained unreliable even after multiple rounds of configuration fixes.
- `bge-reranker` behaved correctly and is the more trustworthy rerank model in the current environment.

## 13. Suggested next steps

If this work continues, the most useful follow-up directions are:

1. Keep `Qwen3-Reranker` behind a dedicated correctness benchmark before using it in production.
2. Add a Transformers-based comparison script for rerank evaluation.
3. Treat rerank model onboarding as a model-specific integration task, not a generic vLLM drop-in.
4. Preserve notes about:
   - `runner` vs `task`
   - special `hf_overrides`
   - model-specific templates
   - whether online and offline scoring match

