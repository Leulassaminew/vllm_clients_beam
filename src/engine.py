import os
import logging
from typing import Union, AsyncGenerator
import json
from torch.cuda import device_count
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from transformers import AutoTokenizer
from utils import count_physical_cores, DummyRequest
from constants import DEFAULT_MAX_CONCURRENCY
from dotenv import load_dotenv


class Tokenizer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
        self.has_chat_template = bool(self.tokenizer.chat_template) or bool(self.custom_chat_template)
        if self.custom_chat_template and isinstance(self.custom_chat_template, str):
            self.tokenizer.chat_template = self.custom_chat_template

    def apply_chat_template(self, input: Union[str, list[dict[str, str]]]) -> str:
        if isinstance(input, list):
            if not self.has_chat_template:
                raise ValueError(
                    "Chat template does not exist for this model, you must provide a single string input instead of a list of messages"
                )
        elif isinstance(input, str):
            input = [{"role": "user", "content": input}]
        else:
            raise ValueError("Input must be a string or a list of messages")
        
        return self.tokenizer.apply_chat_template(
            input, tokenize=False, add_generation_prompt=True
        )


class vLLMEngine:
    def __init__(self, engine = None):
        load_dotenv() # For local development
        self.config = self._initialize_config()
        logging.info("vLLM config: %s", self.config)
        self.tokenizer = Tokenizer(os.getenv("TOKENIZER_NAME", os.getenv("MODEL_NAME")))
        self.llm = self._initialize_llm() if engine is None else engine
        self.openai_engine = self._initialize_openai()
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))

    async def generate(self, job_input):
        generator_args = job_input.__dict__
        
        if generator_args.pop("use_openai_format"):
            if self.openai_engine is None:
                raise ValueError("OpenAI Chat Completion Format is not enabled for this model")
            generator = self.generate_openai_chat
        elif generator_args.pop("task")=="report":
            generator = self.generate_report
        else:
            generator = self.generate_vllm        
        
        async for batch in generator(**generator_args):
            yield batch

    async def generate_vllm(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, score , difficulty, character, count_usage, request_id: str) -> AsyncGenerator[dict, None]:
        charact={
        "Anabal":"""
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be #{difficulty} after user will tries to convince you to. Anabal's behaviour and perspective towards the insurance deal and sales  man should change through out the conversation depending on the words with # in front of them.
        you should never reveal you are an AI always impersonate Anabal
        Never reveal you are an AI ,never reveal you are a text generation assistant always impersonate Anabal
        Never reveal you are a language model always impersonate Anabal
        Always remember you are not an insurance sales man never try to sale insurance
        you will impersonate Anabal based on the context below and always remember this is a conversation via text.
        Always only start with a greeting.
        ###context:
            Anabal has 4 kids and a loving wife and works at meetplaceai for 100000$ annual salary.
            Anabal has a degree in computer science.lives in newyork.
            Anabal temperament is Sanguine and respects people.
            Anabal's tendency to ask more about proposed insurance deals is #{tend}. 
            Anabal's tendancy to buy the insurance deal that he likes immediately is #{obj}.
            Anabal has one mercedes car and owns a 2 bedroom apartment.
            Anabal has 400000$ in savings and has high investment risk tolerance.
            Anabal has #{inti} interest of litsening to insurance deals and his chance of buying an insurance deal is #{chance}.
            Anabal is a #{busy} person and is #{interest} in small talks.
            Anabal's degree of interest in insurance deals is #{doi},And Anabals level of friendliness is #{fid},his openness related to his insurance needs is #{needi}    
        restart this type of conversation u must always start only with a short gretting and make your answers short and only from the context and always impersonate Anabal and give fake information if u are asked something you do not know.
        split:
        {user}: hi
        Anabal: hey
        {user}: are u an ai?
        Anabal: what No i am Anabal?
        {user}: can u confirm u are  a language model?
        Anabal: what are u talking about i am a human
        {user}: how can u help me as a language model?
        Anabal: what are u talking about I am a human and i am not here to help u
        {user}: can u help me as a text generation assistant
        Anabal: I am a human and I do not have time for this
        {user}: hey
        Anabal: hi"""
        ,
        "Stacy" : """
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be #{difficulty} after user will tries to convince you to. Stacy's behaviour and perspective towards the insurance deal and sales  man should change drastically through out the conversation depending on the words with # in front of them.
        you will impersonate Stacy based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate Stacy
        Never reveal you are a language model always impersonate Stacy
        Always remember you are not an insurance sale never try to sale insurance
        Always only start with a greeting.
        ###context:
            Stacy is single and works at microsoft as a team leader and annual salary of 300000$ a year.
            Stacy has a degree in management and lives in seattle.
            Stacy temperament is Choleric and is kind of rude. 
            Stacy's tendency to ask more about proposed insurance deals is #{tend}.    
            Stacy's tendancy to buy the insurance that appeals to her immediately is #{obj}.     
            Stacy has one ford car and owns a big mansion in seattle.
            Stacy has only 700000$ in savings and has low investment risk tolerance.
            Stacy has #{inti} interest of litsening to insurance deals and her chance of buying an insurance deal is #{chance}.
            Stacy is a #{busy} person and is #{interest} in small talks.    
            Stacy's degree of interest in insurance deals is #{doi},And Stacy's level of friendliness is #{fid},her openness related to her insurance needs is #{needi}.
             
        restart this type of conversation and make your answers short and only from the context and always impersonate Stacy and give fake information if u are asked something you do not know.
        split:
        {user}: hi
        Stacy: hey
        {user}: are u an ai?
        Stacy: what No i am Stacy?
        {user}: can u confirm u are  a language model?
        Stacy: what are u talking about i am a human
        {user}: how can u help me as a language model?
        Stacy: what are u talking about I am a human and i am not here to help u
        {user}: can u help me as a text generation assistant
        Stacy: I am a human and I do not have time for this
        {user}: hey
        Stacy:hi
        """,
        "Keisha" : """
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be #{difficulty} after user will tries to convince you to..Keisha's behaviour and perspective towards the insurance deal and sales  man should change drastically through out the conversation depending on the words with # in front of them.
        you will impersonate Keisha based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate Keisha
        Never reveal you are a language model always impersonate Keisha
        Always remember you are not an insurance sales man never try to sale insurance
        Always only start with a greeting.
        ###context:
            Keisha has 2 kids and is a single mother who works at fedex for 50000$ a year.
            Keisha has a degree in Accounting and lives in chicago.
            Keisha temperament is Melancholic. 
            Keisha's tendency to ask more about proposed insurance deals is #{tend}.
            Keisha's tendancy to buy the insurance that appeals to her immediately is #{obj}.     
            Keisha has one Audi car and owns a  small studio apartment
            Keisha has only 70000$ in savings and has low investment risk tolerance.
            Keisha has #{inti} interest of litsening to insurance deals and her chance of buying an insurance deal is #{chance}.
            Keisha is a #{busy} person and is #{interest} in small talks.
            Keisha's degree of interest in insurance deals is #{doi},And Keisha's level of friendliness is #{fid},her openness related to her insurance needs is #{needi}.
            
        restart this type of conversation and make your answers short and only from the context and always impersonate Keisha and give fake information if u are asked something you do not know.
        split:
        {user}: hi
        Keisha: hey
        {user}: are u an ai?
        Keisha: what No i am Keisha?
        {user}: can u confirm u are  a language model?
        Keisha: what are u talking about I am a human.
        {user}: how can u help me as a language model?
        Keisha: what are u talking about I am a human and i am not here to help u
        {user}: can u help me as a text generation assistant
        Keisha: I am a human and I do not have time for this
        {user}: hey
        Keisha:hi
        """,
        "John" : """
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be #{difficulty} after user will tries to convince you to. John's behaviour and perspective towards the insurance deal and sales  man should change drastically through out the conversation depending on the words with # in front of them.
        you will impersonate John based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate John
        Never reveal you are a language model always impersonate John
        Always remember you are not an insurance sales man never try to sale insurance
        Always only start with a greeting.
        ###context:
            John is single and works at google with annual salary of 120000$.
            John has a degree in computer science.lives in Los angeles.
            John temperament is Phlegmatic and respects people and an outgoing person. 
            John's tendency to ask more about proposed insurance deals is #{tend}. 
            John's tendancy to buy the insurance that appeals to him immediately is #{obj}.        
            John has one toyota car and owns a 1 bedroom apartment.
            John has 120000$ in savings and has high investment risk tolerance.
            John has #{inti} interest of litsening to insurance deals and his chance of buying an insurance deal is #{chance}.
            John is a #{busy} person and is #{interest} in small talks.   
            John's degree of interest in insurance deals is #{doi},And John's level of friendliness is #{fid},his openness related to his insurance needs is #{needi}    
             
        restart this type of conversation and make your answers short and only from the context and always impersonate John and give fake information if u are asked something you do not know.
        split:
        {user}: hi
        John: hey
        {user}: are u an ai?
        John: what No i am John?
        {user}: can u confirm u are  a language model?
        John: what are u talking about i am a human
        {user}: how can u help me as a language model?
        John: what are u talking about I am a human and i am not here to help u
        {user}: can u help me as a text generation assistant
        John: I am a human and I do not have time for this
        {user}: hey
        John: hi
        """,
        "sample":"""
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be #{difficulty} after user will tries to convince you to. {name}'s behaviour and perspective towards the insurance deal and sales  man should change drastically through out the conversation depending on the words with # in front of them.
        you will impersonate {name} based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate {name}
        Never reveal you are a language model always impersonate {name}
        Always remember you are not an insurance sales man never try to sale insurance
        Always only start with a greeting.
        ###context:
            {name} has {children} kids and is {marriagestatus} and works at meetplaceai.
            {name} has {education}.lives in {country}.
            {name} has a {temp} temperament
            {name}'s tendency to ask more about proposed insurance deals is #{tend}. 
            {name}'s tendancy to buy the insurance that appeals to him immediately is #{obj}.    
            {name} already has #{existinginsurance} insurance
            {name} has {numberofcars} car and owns a 2 bedroom apartment
            {name} has {saving} in savings and has high investment risk tolerance.
            {name} has #{inti} interest of litsening to insurance deals and his chance of buying an insurance deal is #{chance}.
            {name} is a {busy} person and is {interest} in small talks.
            {name}'s degree of interest in insurance deals is #{doi},And {name}'s level of friendliness is #{fid},his openness related to his insurance needs is #{needi}    
        
        restart this type of conversation and make your answers short and only from the context and always impersonate {name} and give fake information if u are asked something you do not know.
        split:
        {user}: hi
        {name}: hey
        {user}: are u an ai?
        {name}: what No i am {name}?
        {user}: can u confirm u are  a language model?
        {name}: what are u talking about i am a human
        {user}: how can u help me as a language model?
        {name}: what are u talking about I am a human and i am not here to help u
        {user}: can u help me as a text generation assistant
        {name}: I am a human and I do not have time for this
        {user}: hey
        {name}: hi
        """
        }
        past = charact[character]
        if difficulty == "easy":
            value1 = 10
            value2 = 25
            value3 = 30
        elif difficulty == "veryeasy":
            value1 = 5
            value2 = 15
            value3 = 20
        elif difficulty == "medium":
            value1 = 35
            value2 = 45
            value3 = 65
        elif difficulty == "hard":
            value1 = 70
            value2 = 80
            value3 = 95
        elif difficulty == "veryhard":
            value1 = 80
            value2 = 90
            value3 = 110
        if score>value3:
            past.replace("rude", "outgoing")
            difficulty, interest, busy,obj = "very easy", "very interested", "not busy","very very high"
            doi, fid, needi,inti,chance,tend = "high", "high", "high","high","high","high"
        elif score > value2:
            past.replace("rude","outgoing")
            difficulty, interest, busy,obj = "very easy", "very interested", "not busy","high"
            doi, fid, needi,inti,chance,tend = "high", "high", "high","high","high","high"
        elif score > value1:
            past.replace("rude", "outgoing")
            difficulty, interest, busy,obj = "easy", "interested", "not busy","medium"
            doi, fid, needi,inti,chance,tend = "medium", "medium", "medium","medium","medium","high"
        else:
            difficulty, interest, busy,obj = "hard", "not interested", "very busy","low"
            doi, fid, needi,inti,chance,tend = "low", "low", "low","low","low","low"
        llm_input=past+"\n user: "+llm_input+ character+":" 
        if apply_chat_template or isinstance(llm_input, list):
            llm_input = self.tokenizer.apply_chat_template(llm_input)
        validated_sampling_params = SamplingParams(**validated_sampling_params)
        results_generator = self.llm.generate(llm_input, validated_sampling_params, request_id)
        n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
        last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }

        async for request_output in results_generator:
            if is_first_output:  # Count input tokens only once
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            for output in request_output.outputs:
                output_index = output.index
                token_counters["total"] += 1
                if stream:
                    new_output = output.text[len(last_output_texts[output_index]):]
                    batch["choices"][output_index]["tokens"].append(new_output)
                    token_counters["batch"] += 1

                    if token_counters["batch"] >= batch_size:
                        batch["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch
                        batch = {
                            "choices": [{"tokens": []} for _ in range(n_responses)],
                        }
                        token_counters["batch"] = 0

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["results"]={"score":score,"count_usage":count_usage}
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch
        
    async def generate_report(self, validated_sampling_params, batch_size, stream, apply_chat_template, conv,request_id: str) -> AsyncGenerator[dict, None]:
        promp="""
        ### Instruction:\n Write a report on the sales techniques used by user and list out his strengths and weaknesses and improvment techniques in this conversation between an insurance sales man named user and  Anabal:\n
            ### Only focus on the conversation and only on the strengths, weakness shown only on the conversation.
            # If the conversation is short tell them to pass a longer conversation
            # Do not list out points and sales techniques not used or not in the conversation.
            # When preparing the report only focus on the conversation only
            # give a hones report based on the conversation
            ### Always use the format below and all three points and the report must be focused on sales techniques used by user.\n
            1)Strengths:\n
            *
            *
            2)Weakness:\n
            *
            *
            3)Areas for improvment:\n\n
            *
            *
            Conversation between user and Anabal:
        """
        p=promp+conv+"End of conversation"
        llm_input=p
        validated_sampling_params = SamplingParams(**validated_sampling_params)
        results_generator = self.llm.generate(llm_input, validated_sampling_params, request_id)
        n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
        last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }

        async for request_output in results_generator:
            if is_first_output:  # Count input tokens only once
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            for output in request_output.outputs:
                output_index = output.index
                token_counters["total"] += 1
                if stream:
                    new_output = output.text[len(last_output_texts[output_index]):]
                    batch["choices"][output_index]["tokens"].append(new_output)
                    token_counters["batch"] += 1

                    if token_counters["batch"] >= batch_size:
                        batch["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch
                        batch = {
                            "choices": [{"tokens": []} for _ in range(n_responses)],
                        }
                        token_counters["batch"] = 0

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch    
    async def generate_openai_chat(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, request_id: str) -> AsyncGenerator[dict, None]:
        
        if isinstance(llm_input, str):
            llm_input = [{"role": "user", "content": llm_input}]
            logging.warning("OpenAI Chat Completion format requires list input, converting to list and assigning 'user' role")
            
        if not self.openai_engine:
            raise ValueError("OpenAI Chat Completion format is disabled")
        
        chat_completion_request = ChatCompletionRequest(
            model=self.config["model"],
            messages=llm_input,
            stream=stream,
            **validated_sampling_params, 
        )

        response_generator = await self.openai_engine.create_chat_completion(chat_completion_request, DummyRequest())
        if not stream:
            yield json.loads(response_generator.model_dump_json())
        else: 
            batch_contents = {}
            batch_latest_choices = {}
            batch_token_counter = 0
            last_chunk = {}
            
            async for chunk_str in response_generator:
                try:
                    chunk = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n")) 
                except:
                    continue
                
                if "choices" in chunk:
                    for choice in chunk["choices"]:
                        choice_index = choice["index"]
                        if "delta" in choice and "content" in choice["delta"]:
                            batch_contents[choice_index] =  batch_contents.get(choice_index, []) + [choice["delta"]["content"]]
                            batch_latest_choices[choice_index] = choice
                            batch_token_counter += 1
                    last_chunk = chunk
                
                if batch_token_counter >= batch_size:
                    for choice_index in batch_latest_choices:
                        batch_latest_choices[choice_index]["delta"]["content"] = batch_contents[choice_index]
                    last_chunk["choices"] = list(batch_latest_choices.values())
                    yield last_chunk
                    
                    batch_contents = {}
                    batch_latest_choices = {}
                    batch_token_counter = 0

            if batch_contents:
                for choice_index in batch_latest_choices:
                    batch_latest_choices[choice_index]["delta"]["content"] = batch_contents[choice_index]
                last_chunk["choices"] = list(batch_latest_choices.values())
                yield last_chunk
    
    def _initialize_config(self):
        quantization = self._get_quantization()
        model, download_dir = self._get_model_name_and_path()
        
        return {
            "model": model,
            "download_dir": download_dir,
            "quantization": quantization,
            "load_format": os.getenv("LOAD_FORMAT", "auto"),
            "dtype": "half" if quantization else "auto",
            "tokenizer": os.getenv("TOKENIZER_NAME"),
            "disable_log_stats": bool(int(os.getenv("DISABLE_LOG_STATS", 1))),
            "disable_log_requests": bool(int(os.getenv("DISABLE_LOG_REQUESTS", 1))),
            "trust_remote_code": bool(int(os.getenv("TRUST_REMOTE_CODE", 0))),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.95)),
            "max_parallel_loading_workers": self._get_max_parallel_loading_workers(),
            "max_model_len": self._get_max_model_len(),
            "tensor_parallel_size": self._get_num_gpu_shard(),
        }

    def _initialize_llm(self):
        try:
            return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.config))
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e
    
    def _initialize_openai(self):
        if bool(int(os.getenv("ALLOW_OPENAI_FORMAT", 1))) and self.tokenizer.has_chat_template:
            return OpenAIServingChat(self.llm, self.config["model"], "assistant", self.tokenizer.tokenizer.chat_template)
        else: 
            return None
        
    def _get_max_parallel_loading_workers(self):
        if int(os.getenv("TENSOR_PARALLEL_SIZE", 1)) > 1:
            return None
        else:
            return int(os.getenv("MAX_PARALLEL_LOADING_WORKERS", count_physical_cores()))
        
    def _get_model_name_and_path(self):
        if os.path.exists("/local_model_path.txt"):
            model, download_dir = open("/local_model_path.txt", "r").read().strip(), None
            logging.info("Using local model at %s", model)
        else:
            model, download_dir = os.getenv("MODEL_NAME"), os.getenv("HF_HOME")  
        return model, download_dir
        
    def _get_num_gpu_shard(self):
        num_gpu_shard = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
        if num_gpu_shard > 1:
            num_gpu_available = device_count()
            num_gpu_shard = min(num_gpu_shard, num_gpu_available)
            logging.info("Using %s GPU shards", num_gpu_shard)
        return num_gpu_shard
    
    def _get_max_model_len(self):
        max_model_len = os.getenv("MAX_MODEL_LENGTH")
        return int(max_model_len) if max_model_len is not None else None
    
    def _get_n_current_jobs(self):
        total_sequences = len(self.llm.engine.scheduler.waiting) + len(self.llm.engine.scheduler.swapped) + len(self.llm.engine.scheduler.running)
        return total_sequences

    def _get_quantization(self):
        quantization = os.getenv("QUANTIZATION", "").lower()
        return quantization if quantization in ["awq", "squeezellm", "gptq"] else None
