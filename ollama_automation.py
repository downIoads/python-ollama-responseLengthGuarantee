import subprocess
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "\nSystem Prompt Instruction: Only output raw text in your response, as it will be forwarded through a screen reader and only listened to.\n"

def query_ollama(*, user_prompt: str, target_word_amount: int, model: str = "gpt-oss:20b", system_prompt: str = DEFAULT_SYSTEM_PROMPT, iteration: int = 0, prev_response_words_amount: int = 0):
    # print user prompt once, but not if response has to be iteratively expanded upon
    if iteration == 0:
        print(f"Generating response for user prompt:\n\"{user_prompt}\"\n\n")

    outfile = "response.txt"

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

    # put speaker prefix for vibevoice
    response = result.stdout
    if not response.startswith("Speaker 1: "):
        response = "Speaker 1: " + response
    
    current_word_amount = len(response.split())

    # if current response is not longer than previous response terminate
    if prev_response_words_amount > 0:
        if current_word_amount <= prev_response_words_amount:
            Path(outfile).write_text(response, encoding="utf-8")
            print(f"Aborted repeatedly trying to get a longer response but it was not successful.")
            print(f"Response has been saved to {outfile}.")
            exit(1)

    # enforce length of output (if reply is shorter feed it back as another prompt to extend it)
    if current_word_amount < target_word_amount:
        print(f"Current response length is {current_word_amount} < {target_word_amount}, telling model to expand response..")

        # update the system prompt to include the previous response and a one-time notice to extend the previous response
        system_prompt_extend_reply_addition = f"\nYour response so far has been: {response}\nContinue that response to further elaborate on your reply. Ensure to include the full previous response in your extended response"
        if system_prompt_extend_reply_addition not in system_prompt:
            system_prompt += system_prompt_extend_reply_addition

        query_ollama(user_prompt=user_prompt, target_word_amount=target_word_amount, model=model, system_prompt=system_prompt, iteration=iteration+1, prev_response_words_amount=current_word_amount)

    # persist as response.txt
    Path(outfile).write_text(response, encoding="utf-8")
    print(f"Response has been saved to {outfile}.")
    exit(1)


def main():
    user_prompt = "cool facts about capybaras"
    target_word_amount = 1000 # you should generate a response with at least 2000 words
    model = "llama3.1:8b" # gpt-oss:20b

    query_ollama(user_prompt=user_prompt, target_word_amount=target_word_amount, model=model)


main()
