# generate_entropica_dataset.py
import os
import logging
import time
import csv
from pathlib import Path
import re

from openai import AzureOpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ENTROPICA_1024_VOCAB = [
    "[EOS]", "the", "and", "to", "a", "was", "he", "she", "they", "it", "her", "said", "in", "you", "his",
    "with", "that", "so", "of", "day", "but", "had", "on", "one", "for", "there", "very", "little", "time",
    "i", "happy", "is", "not", "big", "saw", "once", "were", "play", "are", "wanted", "upon", "their",
    "him", "girl", "them", "be", "have", "can", "at", "up", "go", "went", "all", "friends", "when", "we",
    "from", "what", "then", "back", "smiled", "mom", "dad", "home", "out", "like", "good", "new", "some",
    "fun", "made", "an", "my", "me", "came", "no", "too", "now", "will", "into", "about", "your", "do",
    "if", "just", "as", "by", "has", "more", "would", "could", "did", "who", "which", "or", "been", "its",
    "how", "than", "after", "over", "two", "first", "other", "way", "off", "old", "because", "any", "much",
    "only", "see", "even", "must", "own", "us", "our", "let", "know", "take", "where", "am", "may", "well",
    "down", "should", "each", "most", "people", "mr", "get", "work", "life", "before", "great", "same",
    "right", "mean", "use", "still", "also", "many", "such", "through", "long", "being", "might", "say",
    "again", "think", "made", "why", "ask", "men", "never", "here", "car", "give", "world", "last", "need",
    "feel", "try", "few", "something", "around", "another", "house", "while", "found", "come", "part",
    "place", "point", "help", "put", "end", "does", "name", "away", "hand", "show", "every", "tell",
    "small", "set", "three", "want", "air", "got", "thought", "mother", "father", "left", "boy", "dog",
    "cat", "friend", "love", "tree", "ball", "red", "blue", "green", "yellow", "bird", "fish", "ran",
    "jumped", "played", "laughed", "cried", "walked", "home", "school", "park", "cake", "ice", "cream",
    "sun", "moon", "star", "flower", "rain", "snow", "hot", "cold", "fast", "slow", "yes", "no", "please",
    "thank", "you", "hello", "goodbye", "morning", "night", "eat", "drink", "sleep", "wake", "run", "jump",
    "sing", "dance", "read", "write", "draw", "paint", "make", "find", "look", "listen", "talk", "hug",
    "kiss", "smile", "sad", "mad", "scared", "brave", "kind", "nice", "mean", "share", "help", "wait",
    "turn", "win", "lose", "try", "best", "pretty", "ugly", "loud", "quiet", "clean", "dirty", "wet", "dry",
    "open", "close", "high", "low", "near", "far", "left", "right", "front", "back", "top", "bottom",
    "inside", "outside", "under", "over", "next", "between", "behind", "together", "alone", "always",
    "sometimes", "never", "today", "tomorrow", "yesterday", "now", "later", "soon", "already", "still",
    "again", "once", "twice", "many", "few", "all", "none", "some", "any", "more", "less", "most", "least",
    "big", "bigger", "biggest", "small", "smaller", "smallest", "good", "better", "best", "bad", "worse",
    "worst", "happy", "happier", "happiest", "sad", "sadder", "saddest", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "this", "that", "these", "those", "who", "what", "where", "when", "why", "how", "which",
    "whose", "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so", "after", "although",
    "because", "before", "if", "once", "since", "though", "unless", "until", "when", "while", "as",
    "by", "in", "of", "on", "to", "with", "at", "from", "into", "onto", "out", "over", "under", "up",
    "down", "off", "on", "again", "away", "back", "forward", "here", "there", "everywhere", "nowhere",
    "somewhere", "anywhere", "home", "school", "park", "store", "city", "country", "world", "sun", "moon",
    "star", "sky", "cloud", "rain", "snow", "wind", "flower", "tree", "grass", "river", "lake", "sea",
    "beach", "mountain", "hill", "valley", "forest", "desert", "island", "road", "path", "street", "house",
    "room", "door", "window", "bed", "chair", "table", "book", "toy", "game", "ball", "doll", "car", "bike",
    "train", "plane", "boat", "ship", "dog", "cat", "bird", "fish", "rabbit", "mouse", "bear", "lion",
    "tiger", "elephant", "monkey", "giraffe", "zebra", "horse", "cow", "pig", "chicken", "duck", "frog",
    "snake", "spider", "bee", "butterfly", "ant", "ladybug", "worm", "squirrel", "deer", "fox", "wolf",
    "mom", "dad", "brother", "sister", "grandma", "grandpa", "uncle", "aunt", "cousin", "baby", "child",
    "friend", "teacher", "doctor", "police", "fireman", "mailman", "farmer", "cook", "artist", "singer",
    "dancer", "player", "hero", "princess", "king", "queen", "prince", "dragon", "castle", "magic", "wish",
    "dream", "love", "hate", "like", "want", "need", "feel", "think", "know", "see", "hear", "touch",
    "taste", "smell", "eat", "drink", "sleep", "wake", "walk", "run", "jump", "swim", "fly", "climb",
    "fall", "sit", "stand", "lie", "laugh", "cry", "smile", "frown", "shout", "whisper", "sing", "dance",
    "read", "write", "draw", "paint", "build", "break", "fix", "clean", "dirty", "open", "close", "start",
    "stop", "begin", "end", "win", "lose", "try", "fail", "succeed", "help", "hurt", "share", "take",
    "give", "receive", "buy", "sell", "find", "lose", "keep", "throw", "catch", "kick", "hit", "hug",
    "kiss", "hold", "carry", "push", "pull", "lift", "drop", "wait", "hurry", "slow", "fast", "early",
    "late", "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night", "week", "month",
    "year", "spring", "summer", "fall", "winter", "hot", "cold", "warm", "cool", "wet", "dry", "bright",
    "dark", "light", "heavy", "loud", "quiet", "soft", "hard", "smooth", "rough", "sweet", "sour", "salty",
    "bitter", "yummy", "yucky", "beautiful", "ugly", "pretty", "handsome", "cute", "scary", "funny",
    "serious", "kind", "mean", "nice", "rude", "brave", "scared", "strong", "weak", "smart", "silly",
    "good", "bad", "happy", "sad", "angry", "calm", "excited", "bored", "tired", "awake", "hungry", "full",
    "thirsty", "sick", "healthy", "big", "small", "tall", "short", "long", "short", "wide", "narrow",
    "thick", "thin", "old", "young", "new", "old", "clean", "dirty", "rich", "poor", "safe", "dangerous",
    "easy", "hard", "fun", "boring", "interesting", "simple", "difficult", "right", "wrong", "true",
    "false", "yes", "no", "maybe", "always", "never", "sometimes", "often", "rarely", "all", "none",
    "some", "many", "few", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # extra common flexible words
    "walked", "walking", "played", "playing", "ran", "running", "jumping", "smiled", "smiling", "laughed",
    "laughing", "cried", "crying", "hugged", "hugging", "kissing", "giving", "gave", "taken", "taking",
    "holding", "held", "opened", "opening", "closed", "closing", "called", "calling", "telling", "told",
    "asking", "asked", "answer", "answered", "answering", "looking", "looked", "watching", "watched",
    "listened", "listening", "heard", "hearing", "feeling", "felt", "thinking", "thought", "knowing",
    "known", "sharing", "shared", "helped", "helping", "stayed", "staying", "sat", "sitting",
    "stood", "standing", "reading", "writing", "drawing", "painting", "building", "built", "fixing",
    "fixed", "cleaning", "brighter", "brightest", "darker", "darkest", "warmer", "warmest", "cooler",
    "coolest", "softer", "softest", "harder", "hardest", "slower", "slowest", "faster", "fastest",
    "earlier", "earliest", "later", "latest", "nearby", "farther", "farthest", "closer", "closest",
    "upstairs", "downstairs", "anytime", "someday", "tonight", "maybe", "really", "quite", "almost",
    "together", "apart", "beside", "between", "behind", "above", "below", "across", "through", "toward",
    "onto", "away", "forward", "middle", "corner", "edge", "center", "circle", "square", "line", "box",
    "bag", "hat", "shirt", "pants", "shoes", "coat", "dress", "blanket", "pillow", "floor", "ceiling",
    "wall", "garden", "yard", "porch", "step", "stairs", "swing", "slide", "sandbox", "tower", "bridge",
    "puddle", "shadow", "noise", "quietly", "softly", "gently", "quickly", "slowly", "suddenly",
    "carefully", "happily", "sadly", "angrily", "bravely", "kindly", "nicely", "safely", "alone",
    "story", "stories", "letter", "letters", "page", "pages", "picture", "pictures", "photo", "photos",
    "map", "maps", "shape", "shapes", "color", "colors", "rock", "rocks", "stone", "stones", "stick",
    "sticks", "leaf", "leaves", "branch", "branches", "pathway", "field", "fields", "farm", "farms",
    "barn", "barns", "market", "markets", "shop", "shops", "kitchen", "bathroom", "hall", "hallway",
    "attic", "basement", "shelf", "shelves", "pocket", "pockets", "window", "windows", "curtain",
    "curtains", "cushion", "cushions", "sofa", "couch", "mirror", "clock", "clocks",
    "balloon", "balloons", "block", "blocks", "puzzle", "puzzles", "rope", "string", "ribbon", "toybox",
    "robot", "robots", "train", "trains", "engine", "engines", "drum", "drums", "guitar", "piano",
    "violin", "flute", "song", "songs", "music", "rhythm", "pattern", "patterns",
    "cookie", "cookies", "candy", "candies", "chocolate", "bread", "butter", "cheese", "soup",
    "sandwich", "sandwiches", "milk", "juice", "water", "tea", "breakfast", "lunch", "dinner", "snack",
    "apple", "apples", "banana", "bananas", "grape", "grapes", "pear", "pears", "peach", "peaches",
    "orange", "oranges", "berry", "berries", "carrot", "carrots", "potato", "potatoes",
    "heroic", "gentle", "worried", "curious", "proud", "lonely", "peaceful", "playful", "friendly",
    "helpful", "honest", "polite", "patient", "lucky", "unlucky", "carefree", "serious", "quiet",
    "shy", "nervous", "surprised", "sleepy", "noisy", "messy", "tidy"
]

# Remove duplicates and limit to 1023 + [EOS]
VOCAB = ["[EOS]"] + list(dict.fromkeys(ENTROPICA_1024_VOCAB[1:]))[:1023]
# VOCAB is the canonical list used for both prompting and tokenization.

PROMPT = f"""
You are a children's book author.

You MUST follow these hard rules with NO exceptions:

1. You may ONLY use the {len(VOCAB)} words from this vocabulary list (case-insensitive):
   {', '.join(VOCAB)}
2. Do NOT invent any new words, names, or spellings that are not in this list.
3. Do NOT change a word into a different form unless that exact form also appears in the list.
   For example, if "run" is in the list but "running" is not, you must NOT write "running".
4. You may ONLY use spaces and the period character "." as punctuation.
   Do NOT use commas, question marks, exclamation marks, quotes, dashes, or any other symbols.
5. Each story should be a very short, simple children's story. Aim for 2–6 sentences per story.
6. If you cannot write a sentence without breaking these rules, you MUST rewrite the sentence
   using only allowed words.

Now write 50,000 very short, simple stories that follow ALL of the rules above.

Examples (these examples follow the rules):
The little cat saw a small bird. The bird was blue. The cat was happy.
The big dog ran to the little boy. The boy gave the dog a red ball. They played all day.
"""

endpoint = "https://damir-mvp-01.openai.azure.com"
deployment = "gpt-4.1-mini"  # your chosen deployment
subscription_key = os.getenv(
    "AZURE_OPENAI_KEY",
    "",
)
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version=api_version,
)


def call_openai_text(messages: list[dict], temperature: float = 0.2) -> str:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    Returns assistant text using the same Azure OpenAI deployment.
    """
    logger.info("[llm_extract] message going to LLM: %s", messages)
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    logger.info("[llm_extract] raw content from LLM: %s", resp.choices[0].message.content)
    return resp.choices[0].message.content or ""


# Output paths
csv_path = Path("entropica_1024_stories.csv")
ids_path = Path("entropica_1024_train_ids.txt")

# Build word -> id mapping once for tokenization (must match the trimmed VOCAB)
word_to_id = {w: i for i, w in enumerate(VOCAB)}

# Explicit ids: period token and OOV token
PERIOD_ID = 0
OOV_ID = 1023

total_stories = 0
skipped_oov = 0

with csv_path.open("w", newline="", encoding="utf-8") as f_csv, ids_path.open(
    "w", encoding="utf-8"
) as f_ids:
    writer = csv.writer(f_csv)
    # TinyStories-style header (we're skipping the header to match your current format)
    # writer.writerow(["text"])

    # 2000 batches; prompt asks for "Batch X of 50 stories:"
    for i in range(2000):   # adjust this if you want a different total target
        print(f"Batch {i + 1}/2000")
        messages = [
            {
                "role": "user",
                "content": PROMPT + f"\n\nBatch {i + 1} of 50 stories:",
            }
        ]
        reply_text = (call_openai_text(messages, temperature=0.0) or "").strip()
        batch = reply_text.split("\n\n")

        for raw in batch:
            s = raw.strip()
            if not s:
                print("Skipping empty chunk...")
                continue

            # Remove leading numbering formats like "1. story", "2) story", "12 story"
            s = re.sub(r"^\s*\d+\s*[\.)-]?\s*", "", s)

            if not s:
                print("Skipping empty after cleaning numbering...")
                continue

            # --- Tokenize first, and detect OOV usage ---
            tokens_int: list[int] = []
            has_oov = False

            # Normalize newlines to spaces and lowercase
            for word in s.replace("\n", " ").split():
                clean = word.lower()

                # Case 1: bare period token
                if clean == ".":
                    tokens_int.append(PERIOD_ID)
                    continue

                # Case 2: word ending with a single period (e.g., "dog.")
                if clean.endswith("."):
                    base = clean[:-1]
                    if base:
                        tok_id = word_to_id.get(base, OOV_ID)
                        if tok_id == OOV_ID:
                            has_oov = True
                        tokens_int.append(tok_id)
                    # always add a period token after sentence-ending word
                    tokens_int.append(PERIOD_ID)
                    continue

                # Case 3: regular word without trailing period
                base = clean
                if not base:
                    continue
                tok_id = word_to_id.get(base, OOV_ID)
                if tok_id == OOV_ID:
                    has_oov = True
                tokens_int.append(tok_id)

            if tokens_int and not has_oov:
                # write human-readable story without numbering
                writer.writerow([s])
                # write tokenized version (space-separated ids)
                tokens_str = " ".join(str(t) for t in tokens_int)
                f_ids.write(tokens_str + "\n")
                total_stories += 1
            elif tokens_int:
                skipped_oov += 1
                print("Skipping story due to OOV token(s).")

        # small pause to be nice to the API
        time.sleep(1)

print(
    f"Generated {total_stories} perfect stories using only {len(VOCAB)} words "
    f"into {csv_path} and {ids_path}. Skipped {skipped_oov} stories due to OOV tokens."
)


