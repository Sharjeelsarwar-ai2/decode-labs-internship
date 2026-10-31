"""
Decode Labs — Project 1: Rule-Based AI Chatbot
------------------------------------------------

Every reply comes from an explicit if-else rule written below.

Type 'bye', 'exit' or 'quit' to stop the chatbot.
"""

import datetime
import random


# A small fixed list of jokes. The bot picks one at random,
# but it can only ever pick a joke from this list — still rule-based.
JOKES = [
    "Why did the developer go broke? Because he used up all his cache!",
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why did the computer go to the doctor? It had a virus!",
]


def normalize_input(user_input):
    """Clean the input so one rule covers many spellings.

    Example: '  HELLO!!! ' -> 'hello'
    Without this, 'Hello', 'HELLO' and 'hello!' would need 3 separate rules.
    """
    text = user_input.lower().strip()
    # Remove trailing punctuation only, so "hello!" and "hello" match.
    text = text.strip(" .!?,")

    return text


def get_response(text, name):
    """Decide one reply for one cleaned input.

    The ORDER of these if-elif checks is important:
    specific rules first, general rules later, fallback last.
    Returns (reply, name) because the bot may learn the user's name.
    """

    # 1. Remember the user's name (state / memory in a variable)
    if "my name is" in text:
        new_name = text.replace("my name is", "").strip().title()
        if new_name:
            return f"Nice to meet you, {new_name}! I will remember your name.", new_name
        return "Nice to meet you! What is your name?", name

    # 2. Greetings
    if text in ("hello", "hi", "hey", "salam", "assalam-o-alaikum", "hello bot"):
        if name:
            return f"Hello, {name}! How can I help you today?", name
        return "Hello! How can I help you today?", name

    # 3. How are you?
    elif "how are you" in text:
        return "I'm doing well, thank you! How can I help you?", name

    # 4. The bot's own name
    elif "what is your name" in text or "who are you" in text:
        return "I'm DecodeBot, a rule-based chatbot built for Decode Labs Project 1.", name

    # 5. Who made the bot?
    elif "who made you" in text or "who created you" in text or "who built you" in text:
        return "I was built by Sharjeel Sarwar as a Decode Labs AI internship project.", name

    # 6. About Decode Labs
    elif "decode labs" in text:
        return "Decode Labs is where I was built — for an Artificial Intelligence internship.", name  
    # 7. Ask the bot if it knows your name
    elif "what is my name" in text or "do you know my name" in text:
        if name:
            return f"Yes, your name is {name}.", name
        return "I don't know your name yet. What is your name?", name

    # 8. Help / what the bot can do
    elif "help" in text or "what can you do" in text:
        return (
            "Try: hello, how are you, what is your name, time, date, "
            "tell me a joke, my name is ..., or bye to exit."
        ), name

    # 9. Current time (real time from the computer clock, not a fixed reply)
    elif "what is the time" in text:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}.", name

    # 10. Current date
    elif "what is the date today" in text or "what day is it" in text:
        today = datetime.datetime.now().strftime("%A, %d %B %Y")
        return f"Today is {today}.", name

    # 11. A joke
    elif "tell me a joke" in text:
        return random.choice(JOKES), name

    # 12. Thank you
    elif "thank you" in text:
        return "You're welcome! Happy to help.", name

    # 13. Fallback — runs ONLY if no rule above matched.
    # This must stay last, otherwise it would block every other rule.
    else:
        return "Sorry, I didn't understand that. Type 'help' to see what I can do.", name


def main():
    """Run the continuous chat loop.

    while True keeps asking forever.
    The loop only stops when the user types an exit word -> break.
    """
    print("DecodeBot: Hello! I'm a rule-based chatbot. Type 'bye' to exit.")
    name = None  # No name known yet; get_response() may fill this in.

    while True:
        user_input = input("You: ")
        text = normalize_input(user_input)

        # Empty input: don't crash, just ask again (continue = next loop round)
        if not text:
            print("DecodeBot: Please type something.")
            continue

        # Exit commands: print goodbye and break out of the loop
        if text in ("bye", "bye bye", "goodbye", "good bye", "exit", "quit", "stop"):
            print("DecodeBot: Goodbye! It was nice talking to you.")
            break

        # Normal case: decide a reply and print it, then loop again
        reply, name = get_response(text, name)
        print(f"DecodeBot: {reply}")


if __name__ == "__main__":
    main()
