import time
from .base import BaseLLM
import base64
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

class OpenAILLM(BaseLLM):

    def __init__(self, model_name='', system_message=None, api_key=None, base_url=None, timeout=60.0, max_retries=3):
        super(OpenAILLM, self).__init__(model_name, system_message, api_key, base_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.max_retries = max_retries

    def _retry_api_call(self, call_func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return call_func(*args, **kwargs)
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                print(f'\n[API warning] Request failed (attempt {attempt + 1}/{self.max_retries}): {type(e).__name__} - {e}')
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt
                    print(f' -> Retrying in {sleep_time} s ...')
                    time.sleep(sleep_time)
                else:
                    print('[API error] Max retries exceeded; giving up.')
                    raise e
            except Exception as e:
                raise e

    def chat(self, prompt):
        self.conversation_history.append({'role': 'user', 'content': prompt})

        def make_request():
            return self.client.chat.completions.create(model=self.model_name, messages=self.conversation_history, temperature=0.85, top_p=0.95, stream=False)
        response = self._retry_api_call(make_request)
        bot_response = response.choices[0].message.content
        self.conversation_history.append({'role': 'assistant', 'content': bot_response})
        return bot_response

    def predict(self, prompt, context=None):
        messages = []
        if context:
            messages.append({'role': 'system', 'content': context})
        messages.append({'role': 'user', 'content': prompt})

        def make_request():
            return self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.85, top_p=0.95, stream=False)
        response = self._retry_api_call(make_request)
        return response.choices[0].message.content

    def clear_history(self):
        self.conversation_history = []

    def evaluate_audio(self, audio_path, prompt):
        with open(audio_path, 'rb') as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        ext = audio_path.split('.')[-1].lower()
        audio_format = ext if ext in ['wav', 'mp3'] else 'wav'
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': f'data:audio/{audio_format};base64,{encoded_audio}'}}]}]

        def make_request():
            return self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.85, timeout=180.0)
        response = self._retry_api_call(make_request)
        return response.choices[0].message.content
