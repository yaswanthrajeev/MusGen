import modal
import os
import uuid
from pydantic import BaseModel
import requests
import base64
import boto3
from typing import List
from prompts import PROMPT_GENERATOR_PROMPT, LYRICS_GENERATOR_PROMPT
app = modal.App("music-generator")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", "cd /tmp/ACE-Step && pip install ."])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name("ace-steps-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music-gen-secret")

class AudioGenerationBase(BaseModel):
    audio_duration: float = 180.0
    seed: int= -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False

class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str

class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str
    lyrics: str

class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str
        
class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]

class GenerateMusicResponse(BaseModel):
    audio_data: str


@app.cls(image=image,
        gpu = "L40S",
        volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
        secrets=[music_gen_secrets],
        scaledown_window=15
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from diffusers import AutoPipelineForText2Image
        import torch
        import uuid
        ##Music gen model
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16" ,
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        ##llm model
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )
    ## stable diffusion model
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16",cache_dir="/.cache/huggingface" )
        self.image_pipe.to("cuda")

    def prompt_qwen(self, question: str):
        messages = [
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_prompt(self, description: str):
        #insert description into templates
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt = description)
        return self.prompt_qwen(full_prompt)
        # run llm inference and return chats
        
    def generate_lyrics(self, description: str):
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description = description)
        return self.prompt_qwen(full_prompt)

    def generate_categories(self, description: str)->List[str]:
        prompt = f"Based on the following music description, list 3-5 relevant genres or categories as a comma-sepeated list. For example: Pop, Electronic, Sad, 80s. Description:'{description}'"
        response_text = self.prompt_qwen(prompt)
        categories = [cat.strip()
                      for cat in response_text.split(",") if cat.strip()]
        return categories

    def generate_and_upload_to_s3(
        self,
        prompt: str,
        lyrics: str,
        instrumental: bool,
        audio_duration: float,
        infer_step: int,
        guidance_scale: float,
        seed: int,
        description_for_categorization: str
    )-> GenerateMusicResponseS3:
        final_lyrics = "[instrumental]" if instrumental else lyrics
        print(f"Generated lyrics: \n{final_lyrics}")
        print(f"prompt: \n{prompt}")

        #aws
        #AKIA35Q4E4G3UWKBQP4T - Active
        #create s3 bucket:thumbnail,songs
        #iam user:
        # - Frontend user(nextjs):getting objects from s3 objects: GetObject,ListObject
        #- Backedn user(modal): upload to buvket:PutObject,Get and List
        s3_client = boto3.client("s3")
        bucket_name = os.environ["S3_BUCKET_NAME"]

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir,exist_ok=True)
        output_path = os.path.join(output_dir,f"{uuid.uuid4()}.wav")
        self.music_model(
                prompt=prompt,
                lyrics=final_lyrics,
                audio_duration= audio_duration,
                infer_step = infer_step,
                guidance_scale = guidance_scale,
                save_path = output_path,
                manual_seeds = str(seed)
        )

        audio_s3_key = f"{uuid.uuid4()}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        os.remove(output_path)

        #thumbnail
        thumbnail_prompt = f"{prompt}, album cover art"
        image = self.image_pipe(prompt=thumbnail_prompt, num_inference_steps=2, guidance_scale=0.0).images[0] 

        image_output_path = os.path.join(output_dir,f"{uuid.uuid4()}.png")
        image.save(image_output_path)
        image_s3_key = f"{uuid.uuid4()}.png"
        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)

        #category generation: "hi hop,"rock"


        categories = self.generate_categories(description_for_categorization)

        return GenerateMusicResponseS3(
            s3_key = audio_s3_key,
            cover_image_s3_key = image_s3_key,
            categories = categories
        )



    @modal.fastapi_endpoint(method="POST", requires_proxy_auth = True)
    def generate(self)-> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir,exist_ok=True)
        output_path = os.path.join(output_dir,f"{uuid.uuid4()}.wav")
        self.music_model(
                prompt="country rock, folk rock, southern rock, bluegrass, country pop",
                lyrics="[verse]\nWoke up to the sunrise glow\nTook my heart and hit the road\nWheels hummin' the only tune I know\nStraight to where the wildflowers grow\n\n[verse]\nGot that old map all wrinkled and torn\nDestination unknown but I'm reborn\nWith a smile that the wind has worn\nChasin' dreams that can't be sworn\n\n[chorus]\nRidin' on a highway to sunshine\nGot my shades and my radio on fine\nLeave the shadows in the rearview rhyme\nHeart's racing as we chase the time\n\n[verse]\nMet a girl with a heart of gold\nTold stories that never get old\nHer laugh like a tale that's been told\nA melody so bold yet uncontrolled\n\n[bridge]\nClouds roll by like silent ghosts\nAs we drive along the coast\nWe toast to the days we love the most\nFreedom's song is what we post\n\n[chorus]\nRidin' on a highway to sunshine\nGot my shades and my radio on fine\nLeave the shadows in the rearview rhyme\nHeart's racing as we chase the time",
                audio_duration= 220,
                infer_step = 60,
                guidance_scale = 15,
                save_path = output_path
        )
        with open(output_path, "rb") as f: ## rb = read binary
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        os.remove(output_path)

        return GenerateMusicResponse(audio_data = audio_b64)



    @modal.fastapi_endpoint(method="POST", requires_proxy_auth = True)
    def generate_from_description(self, request: GenerateFromDescriptionRequest)-> GenerateMusicResponseS3:
        ## Generating a prompt
        prompt = self.generate_prompt(request.full_described_song)

        # Generate lyrics
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.full_described_song)

        return self.generate_and_upload_to_s3(prompt=prompt, lyrics = lyrics,
                                               description_for_categorization=request.full_described_song, **request.model_dump(exclude={"full_described_song"}))

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth = True)
    def generate_with_lyrics(self,request:GenerateWithCustomLyricsRequest)-> GenerateMusicResponseS3:
                return self.generate_and_upload_to_s3(
                                               description_for_categorization=request.prompt, **request.model_dump())


    @modal.fastapi_endpoint(method="POST", requires_proxy_auth = True)
    def generate_with_described_lyrics(self, request: GenerateWithDescribedLyricsRequest)-> GenerateMusicResponseS3:
           # Generate lyrics
                lyrics=""
                if not request.instrumental:
                    lyrics = self.generate_lyrics(request.described_lyrics)
                return self.generate_and_upload_to_s3(prompt=request.prompt, lyrics = lyrics,
                                               description_for_categorization=request.prompt, **request.model_dump(exclude={"described_lyrics","prompt"}))




@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate_with_described_lyrics.get_web_url()

    request_data = GenerateWithDescribedLyricsRequest(
        prompt="melody, 180bpm, drums, electronic",
        described_lyrics="a lyrics about love and lonliness",
        guidance_scale = 15

    )



    payload = request_data.model_dump()
    response = requests.post(endpoint_url, json = payload)
    response.raise_for_status()
    result = GenerateMusicResponseS3(**response.json())

    print(f"Success: {result.s3_key}  {result.cover_image_s3_key}  {result.categories}")
   #audio_bytes = base64.b64decode(result.audio_data)
    #output_filename = "generated.wav"
    #with open(output_filename, "wb") as f:
     #   f.write(audio_bytes)

