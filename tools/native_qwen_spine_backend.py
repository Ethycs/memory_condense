"""Local Qwen generation on date-neutral, summary-only native spine jobs."""
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_merges import neutral_key, neutral_messages
from tools.assemble_native_spine_summaries import digest
from tools.local_qwen_spine_backend import LocalQwenBackend, decode_rows


class NativeQwenBackend(LocalQwenBackend):
    max_batch_size = 4

    def __init__(self, *args):
        super().__init__(*args)
        self.identity.update({
            "native_adapter_sha256": digest(__file__),
            "native_messages_sha256": digest("src/memory_condense/search/native_spine_merges.py"),
            "timestamp_metadata_in_model_inputs": False,
        })
        self.identity_sha256 = identity_sha256(self.identity)

    def generate(self, jobs, attempt):
        jobs = tuple(jobs)
        if not 1 <= len(jobs) <= self.max_batch_size:
            raise ValueError("native local batch must contain one to four summary jobs")
        messages = [neutral_messages(job, attempt) for job in jobs]
        keys = [neutral_key(job) for job in jobs]
        if len(set(keys)) != len(keys):
            raise ValueError("deduplicate identical native merge inputs before generation")
        self.load()
        import torch
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for m in messages]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        width = inputs["input_ids"].shape[1]
        if width > self.identity["max_input_tokens"]:
            raise ValueError("native summary job exceeds the local input budget")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        eos = self.model.generation_config.eos_token_id
        rows = decode_rows(self.tokenizer, output, width, {eos} if isinstance(eos, int) else set(eos))
        for row, key, tokens in zip(rows, keys, inputs["attention_mask"].sum(dim=1), strict=True):
            row.update(merge_key=key, input_tokens=int(tokens))
        return {"backend_sha256": self.identity_sha256, "rows": rows, "elapsed_s": elapsed,
                "peak_gpu_allocated_GiB": torch.cuda.max_memory_allocated()/2**30,
                "raw_inputs_to_qwen": False, "timestamp_metadata_in_model_inputs": False,
                "remote_provider_calls": 0}
