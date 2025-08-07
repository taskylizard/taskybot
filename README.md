## taskybot

> [!IMPORTANT]
> This is just a code dump for learning purposes. You'll probably want to modify this for production use.

State of the art™ shitposting technology made by the cringe.

This uses my discord data request to scrape messages, fine-tune a model, and then use it to respond to mentions in whitelisted discord channels.

You will need to request your Discord data for this. You can do this by going to your Discord settings, Data & Privacy, and clicking "Export Data", selecting only "Messages". It may take a day or two depending on your account.

### Corpus generation and jank and jank and shit

The scraper is a bit of a hack, but it works. It excludes messages with certain keywords, and only includes messages from a certain time period (`cutoff_days` in `scrape.py`), processes them into the OpenAI chat format (`user`/`assistant`/`system`). It runs on the `Messages` directory of the data export, and looks for a channel ID like `Messages/c{CHANNEL_ID}/messages.json`. Finally after much preprocessing, it saves the converations to a JSON file in the `/corpus` folder.

### Fine-tuning Magic Voodoo Work I Don't Understand

Next, I use the conversations to fine-tune a language model. I chose Llama 3.1 because of its permissive license (haha no) and high quality relative to its small size. Fine-tuning is done using Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique that produces a small adapter that can be merged with the base model when needed.

The fine-tuning implementation uses torchtune, a PyTorch library for easily configuring fine-tuning runs.

### Inference work

I use vLLM to run the fine-tuned model. It supports LoRA adapters out of the box. It can remain fast at `~25 tok/s` on a single A100 GPU, thanks to prefix caching and keeping its cache warm on a hard volume. Important variables for that are: `enforce_eager` disables both Torch compilation and CUDA graph capture, `enable_prefix_caching` allows the vLLM engine to reuse cached KV (key-value) pairs from previous prompts if a new query shares the same
prefix, and `tensor_parallel_size` is set to assume multiple GPUs are for splitting up large matrix multiplications.

See the `inference.py` file for the code that runs the model. It's not much to look at, but it's a good example of how to use vLLM progmatically.

### Other stuff

I use Weights and Biases for logging and tracking fine-tuning runs. HuggingFace Hub for the model, and discord.py for the bot. You will need tokens for all of these.

Meta also requires you to sign an agreement to use their model. It's a bit of a pain, but there are other models that are open anyway you can switch to.
