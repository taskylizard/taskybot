import time
from typing import AsyncIterator
from .common import (
    get_user_checkpoint_path,
    output_vol,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

MINUTES = 60  # seconds


class Inference:
    def enter(self):
        engine_args = AsyncEngineArgs(
            model="/vol/model",
            gpu_memory_utilization=0.95,
            tensor_parallel_size=1,
            enable_lora=True,
            enforce_eager=True,
            max_lora_rank=32,
            max_model_len=4096,
            max_loras=16,
            enable_prefix_caching=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.loras: dict[str, int] = dict()  # per replica LoRA identifier

    async def generate(self, inpt: list[dict], user: str) -> AsyncIterator[str]:
        if (ident := user) not in self.loras:
            self.loras[ident] = len(self.loras) + 1
            output_vol.reload()
        checkpoint_path = get_user_checkpoint_path(user)
        lora_request = LoRARequest(
            ident, self.loras[ident], lora_local_path=checkpoint_path
        )

        tokenizer = await self.engine.get_tokenizer(lora_request=lora_request)

        prompt = tokenizer.apply_chat_template(
            conversation=inpt, tokenize=False, add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=1,
            top_p=0.95,
            top_k=50,
            max_tokens=2048,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            tokens = len(request_output.outputs[0].token_ids)

        throughput = tokens / (time.time() - t0)
        print(f"🧠: Effective throughput of {throughput:.2f} tok/s")
