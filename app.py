import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import gradio as gr
from peft import PeftModel
import re
import hashlib
import io

model_id = "hitmanonholiday/llava-1.5-7b-4bit-finetuned3"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                      quantization_config=quantization_config,
                                                      torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. \
                        The assistant gives helpful, detailed, and polite answers to the user's questions. \
                        {% for message in messages %}{% if message['role'] == 'user' %}\
                        USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}\
                        {% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""


tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
processor.tokenizer = tokenizer


class Chat:
  def __init__(self,processor,model):
      self.INSTRUCTION = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
                        """
      self.prompt = self.INSTRUCTION
      self.first_message = None
      self.image = None
      self.processor = processor
      self.model = model
      self.md5_hash = None
  def send_user_message(self,image,message):
      if image != None:
        hash = self.md5_hash_pil_image(image)
        if self.md5_hash != hash:
            self.image = image
            self.prompt = self.INSTRUCTION
            self.first_message = message
            self.md5_hash = hash

      self.prompt = self.prompt + " USER: "+message
      self.prompt = self.prompt.replace(self.first_message, self.first_message+"<image>")
      if self.image == None:
        return "Please upload an image"
      response = self.predict(image = self.image, text = self.prompt)
      self.prompt,flag = self.format_chat(response)
      if flag:
        return "The ASSISTANT COULD NOT UNDERSTAND THE LAST PROMPT\n"+self.prompt.replace(self.INSTRUCTION.strip(),"")
      return self.prompt.replace(self.INSTRUCTION.strip(),"")

  def md5_hash_pil_image(self,img):
    byte_stream = io.BytesIO()
    img.save(byte_stream, format="PNG")  # Save in the original format
    md5_hash = hashlib.md5(byte_stream.getvalue()).hexdigest()
    return md5_hash

  def format_chat(self,text):
      pattern = r'(?=USER:|ASSISTANT:)'
      result = re.split(pattern, text)
      clean_text = [segment.strip() for segment in result]
      text = ""
      for x in clean_text:
        text = text+"\n\n"+x
      return self.remove_last_line_if_no_assistant(text)

  def remove_last_line_if_no_assistant(self,text):
      lines = text.split('\n')
      flag = False
      if lines and "ASSISTANT" not in lines[-1]:
          lines.pop()
          flag = True
      return '\n'.join(lines),flag

  def predict(self, image, text):
      inputs = self.processor(images=image, text=text, return_tensors="pt")
      output = self.model.generate(**inputs, max_new_tokens=100)
      response = self.processor.decode(output[0], skip_special_tokens=True)
      return response

inputs = [
    gr.Image(type="pil", label="Upload an image"),
    gr.Textbox(lines=2, placeholder="Type your text here...", label="Input Text")
    
]

outputs = gr.Textbox(label="Output")

chat = Chat(processor,model)
predict = chat.send_user_message

gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="LLAVA Multimodal Chatbot").launch(share=True)

# Format the text     