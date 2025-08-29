import re
import subprocess
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "\nSystem Prompt Instruction: Only output raw text in your response, as it will be forwarded through a screen reader and only listened to.\n"

# clean_response enforces a certain output:
#       No duplicate sentences allowed
#       Contains only certain chars
#       Starts with 'Speaker 1:' (vibevoice requirement)
def clean_response(*, response: str) -> str:
    # Phase 1: Remove duplicate sentences
    sentences = re.findall(r'[^.?!]+[.?!]', response)
    response = " ".join(dict.fromkeys(s.strip() for s in sentences))

    # Phase 2: Remove illegal chars
    enforced_prefix = "Speaker 1: "
    response = re.sub(r'[^a-zA-Z0-9,.:()\n\- ]', '', response) # only allows a-z, A-Z, 0-9, ',', '.', '(', ')', ' ', '\n', '-'
    
    # Phase 3: Enforce prefix
    if not response.startswith(enforced_prefix):
        response = enforced_prefix + response

    return response

def persist_response(*, response: str, output_filename: str = "response.txt"):
    # write response to file
    Path(output_filename).write_text(response, encoding="utf-8")
    print(f"Response has been saved to {output_filename}.")


def query_ollama(*, user_prompt: str, target_word_amount: int, model: str = "gpt-oss:20b", system_prompt: str = DEFAULT_SYSTEM_PROMPT, iteration: int = 0, prev_response_words_amount: int = 0):
    # print user prompt once, but not if response has to be iteratively expanded upon
    if iteration == 0:
        print(f"Generating response for user prompt:\n\"{user_prompt}\"\n\n")

    result = subprocess.run(
        [   "ollama", 
            "run", 
            model, 
            "--hidethinking", # remove thoughts from output
        ],
        input=user_prompt+system_prompt,
        capture_output=True,
        text=True,
        check=True
    )
    response = result.stdout

    # enforce response expectation
    response = clean_response(response=response)
    
    # if current response is not longer than previous response terminate
    current_word_amount = len(response.split())
    if prev_response_words_amount > 0:
        if current_word_amount < prev_response_words_amount: # do not use <=, it often struggles to find anything new and requires multiple retries
            print(f"Aborted repeatedly trying to get a longer response because the response got shorter.")
            persist_response(response=response, output_filename="response.txt")
            return response

    # enforce length of output (if reply is shorter feed it back as another prompt to extend it)
    if current_word_amount < target_word_amount:
        print(f"Current response length is {current_word_amount} < {target_word_amount}, telling model to expand response..")

        # update the system prompt to include the previous response and a one-time notice to extend the previous response
        system_prompt_extend_reply_addition = f"\nYour response so far has been: {response}\nContinue that response to further elaborate on your reply. Ensure to include the full previous response in your extended response and do not inform the user that are you are continuing a previous response."
        if system_prompt_extend_reply_addition not in system_prompt:
            system_prompt += system_prompt_extend_reply_addition

        response = query_ollama(user_prompt=user_prompt, target_word_amount=target_word_amount, model=model, system_prompt=system_prompt, iteration=iteration+1, prev_response_words_amount=current_word_amount)

    # persist response on disk
    persist_response(response=response, output_filename="response.txt")
    
    return response


def main():
    user_prompt = "cool facts about capybaras"
    target_word_amount = 1000 # generate a response with at least this many words
    model = "llama3.1:8b" # gpt-oss:20b

    response = query_ollama(user_prompt=user_prompt, target_word_amount=target_word_amount, model=model)
    print("\nFinal Response:\n", response)


main()
