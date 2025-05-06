# ----------------------------------
import json
from datetime import datetime
from collections import deque
import random # We'll use this for varied responses later
import torch
from transformers import AutoTokenizer, BlenderbotForConditionalGeneration
import os
import traceback # Import traceback for better error reporting

from transformers import pipeline

# Placeholder for Hugging Face imports - add later

# <<< CHANGE 1: Define paths relative to the script location >>>
# Get the directory where the script (*.py file) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define PROFILE_FILE and LTM_FILE in the same directory as the script
PROFILE_FILE = os.path.join(script_dir, 'profile.json')
LTM_FILE = os.path.join(script_dir, 'long_term_memory.json')
print(f"--- Script Directory: {script_dir}")
print(f"--- Profile File Path: {PROFILE_FILE}")
print(f"--- LTM File Path: {LTM_FILE}")


# profile = {} # Global dictionary to hold profile data - Initialized later in __main__

# --- Moved Profile Loading to __main__ to avoid potential early errors ---
# def load_profile(filepath): ... # Function definition remains the same

# --- Configuration ---
# <<< CHANGE 1: Reduce history length to prevent IndexError >>>
STM_MAXLEN = 2 # Store last 3 exchanges (Try 2 or 4 if 3 still causes issues/lacks context)

# --- Global State (Initialize after model loading or in __main__) ---
profile = {}
short_term_memory = deque(maxlen=STM_MAXLEN)
long_term_memory = []

# --- Model Placeholders (Loaded in __main__) ---
emotion_classifier = None
conversational_tokenizer = None
conversational_model = None
device = None # To store 'cuda' or 'cpu'


def load_profile(filepath):
    """Loads the elder's profile from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Added encoding
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Profile file not found at {filepath}")
        return {} # Return empty dict if file missing
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return {}
    except Exception as e: # Catch other potential errors
        print(f"Error loading profile file {filepath}: {e}")
        return {}

# --- Moved initial profile load check into __main__ ---

def detect_emotion(text):
    """Detects emotion using the pre-loaded pipeline."""
    global emotion_classifier
    if not text or not isinstance(text, str):
        return 'NEUTRAL' # Handle empty or non-string input

    if emotion_classifier is None:
        # print("Info: Emotion classifier not loaded or failed to load.") # Less alarming than Error
        return 'NEUTRAL'

    try:
        results = emotion_classifier(text)
        if results and isinstance(results, list) and results[0] and 'label' in results[0]:
             label = results[0]['label'].upper()
             # Simplify mapping based on common Hugging Face pipeline outputs
             if label == 'POSITIVE' or label == 'LABEL_1':
                 return 'POSITIVE'
             elif label == 'NEGATIVE' or label == 'LABEL_0':
                 return 'NEGATIVE'
             else: # Handle 'NEUTRAL' or other labels explicitly if the model supports them
                 return 'NEUTRAL'
        else:
            # print(f"Warning: Unexpected result format from emotion classifier: {results}") # Debugging only
            return 'NEUTRAL'
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        return 'NEUTRAL' # Default to neutral on error


# Phase 2: Step 5 - Implement Conversational Model (BlenderBot)
# ==============================================================================

def generate_blenderbot_response(user_input, history_deque):
    """Generates a response using BlenderBot, incorporating conversation history."""
    global conversational_model, conversational_tokenizer, device, profile # Added profile for placeholder replacement

    if conversational_model is None or conversational_tokenizer is None:
        print("Error: Conversational model or tokenizer not loaded.")
        return "I'm sorry, I'm having trouble thinking right now."

    # 1. Format History
    history_string = ""
    for exchange in list(history_deque):
        user_msg = exchange.get('user', '')
        ai_msg = exchange.get('ai', '')
        # Basic format, ensure newline separation
        history_string += f"User: {user_msg}\nAI: {ai_msg}\n"

    # 2. Combine history with the new user input
    prompt = history_string + f"User: {user_input}"
    # print(f"\n[Debug] Prompt length (chars): {len(prompt)}") # Optional: monitor prompt length
    # print(f"[Debug] Prompt sent to BlenderBot:\n---\n{prompt}\n---\n") # Uncomment for full prompt debugging

    # 3. Tokenize the prompt
    try:
        # Ensure max_length is not greater than the model's capability
        # Using model's config value directly if available, otherwise default (e.g., 512)
        tokenizer_max_len = getattr(conversational_model.config, 'max_position_embeddings', 512)

        inputs = conversational_tokenizer(
            prompt,
            return_tensors='pt',
            max_length=tokenizer_max_len, # Use model's max length
            truncation=True # Crucial: Truncates if longer than max_length
        )

        # <<< CHANGE 2: Add Tokenized Length Debug Print >>>
        input_ids_length = inputs['input_ids'].shape[1]
        print(f"\n--> [DEBUG] Tokenized input_ids length: {input_ids_length} (Model max: {tokenizer_max_len})\n")
        # Check if it somehow exceeds the expected max length (shouldn't with truncation=True)
        if input_ids_length > tokenizer_max_len:
             print(f"!!! [DEBUG] WARNING: Tokenized length ({input_ids_length}) exceeds model max ({tokenizer_max_len}) DESPITE truncation=True!")
        # <<< END DEBUG PRINT >>>

    except Exception as e:
        print(f"Error during tokenization: {e}")
        traceback.print_exc() # Print full traceback for tokenization errors
        return "I'm sorry, I couldn't process that input."

    # 4. Move inputs to the correct device
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        print(f"Error moving inputs to device {device}: {e}")
        return "I'm having a technical difficulty."

    # 5. Generate Response using the model
    try:
            # Use generation config if available, otherwise defaults
            generation_kwargs = {
                "max_length": 128,
                "min_length": 10,
                "do_sample": True,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "no_repeat_ngram_size": 3,
            }
            # Try loading generation config from model directory if it exists
            try:
                 gen_conf_path = os.path.join(conversational_model.config.name_or_path, 'generation_config.json')
                 if os.path.exists(gen_conf_path):
                     # Note: from_pretrained can load generation_config automatically if present usually
                     # but we can explicitly pass it or merge settings if needed.
                     # For now, we'll rely on the defaults above unless specific override is needed.
                     pass
                     # print("[Debug] Found generation_config.json, consider loading/merging its settings.")
            except:
                 pass # Ignore if path doesn't exist or other issues

            reply_ids = conversational_model.generate(
                **inputs,
                **generation_kwargs
            )
    except IndexError as idx_err:
         print(f"\n!!! IndexError during model generation !!!")
         print(f"Error Details: {idx_err}")
         print("This often means the input length exceeds the model's positional embedding capacity.")
         print(f"Current tokenized input length was: {input_ids_length}")
         print(f"Model's max position embeddings: {getattr(conversational_model.config, 'max_position_embeddings', 'N/A')}")
         print(f"Consider further reducing STM_MAXLEN (currently {STM_MAXLEN}).")
         traceback.print_exc()
         return "I'm sorry, I got confused by the length of our chat. Could you repeat that?"
    except Exception as generation_exception:
            print(f"\n!!! An Exception occurred during model generation !!!")
            print(f"Error Type: {type(generation_exception)}")
            print(f"Error Details: {generation_exception}")
            traceback.print_exc()
            return "I'm sorry, something went wrong while I was thinking about that."


    # 6. Decode the response
    try:
        response = conversational_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error decoding response: {e}")
        return "I understood, but couldn't phrase my reply."

    # 7. Cleanup and Return
    response = response.strip()
    if response.startswith("AI:"):
        response = response[3:].strip()
    elif response.startswith("System:"):
         response = response[7:].strip()

    # <<< CHANGE 3: Add Placeholder Replacement >>>
    # Replace literal "[USER_NAME]" if the fine-tuned model outputs it
    response = response.replace("[USER_NAME]", profile.get('name', 'friend')) # Use 'friend' as fallback

    # print(f"[Debug] Raw AI Response (after cleanup): {response}") # Debugging only
    return response



# Phase 2: Step 6 - Implement Memory System (STM & LTM)
# ==============================================================================

# --- Short-Term Memory (STM) ---
def update_short_term_memory(user_input, ai_response):
    """Adds the latest exchange to the STM deque."""
    global short_term_memory
    try:
        # Ensure inputs are strings
        user_input_str = str(user_input) if user_input is not None else ""
        ai_response_str = str(ai_response) if ai_response is not None else ""
        short_term_memory.append({'user': user_input_str, 'ai': ai_response_str})
        # print(f"[Debug] STM Updated. Size: {len(short_term_memory)}") # Uncomment for debug
    except Exception as e:
        print(f"Error updating short-term memory: {e}")

# --- Long-Term Memory (LTM) ---
def load_ltm(filepath):
    """Loads LTM from JSON file."""
    try:
        if not os.path.exists(filepath):
             # Create the directory if it doesn't exist
             os.makedirs(os.path.dirname(filepath), exist_ok=True)
             with open(filepath, 'w', encoding='utf-8') as f:
                 json.dump([], f)
             print(f"Info: LTM file not found at {filepath}, created empty file.")
             return []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip(): # Handle empty or whitespace-only file
                return []
            return json.loads(content)
    except FileNotFoundError:
        print(f"Info: LTM file not found at {filepath}, starting fresh.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from LTM file {filepath}. Check file content. Starting fresh.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred loading LTM: {e}")
        return []





def save_ltm(filepath, memory_list):
    """Saves LTM list to JSON file."""
    try:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_list, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving LTM to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred saving LTM: {e}")





def check_and_update_long_term_memory(user_input):
    """Checks user input for LTM triggers and updates LTM list if needed."""
    global long_term_memory, profile # Added profile for context
    updated = False

    # Define triggers and keywords (can be expanded)
    triggers = ["remember this", "important:", "don't forget", "note that", "key point is"]
    keywords = ["birthday", "anniversary", "appointment", "doctor", "visit", "travel", "plan", "wedding", "funeral", "born", "died", "medication", "prescription"]

    # Add family names from profile as keywords if available
    family_names = []
    if isinstance(profile.get('family'), dict):
         family_names = [v.get('name', '').lower() for k, v in profile.get('family', {}).items() if isinstance(v, dict) and v.get('name')]
         keywords.extend([name for name in family_names if name]) # Add non-empty lowercase names


    found_memory_details = None
    lower_input = user_input.lower()

    # 1. Check for Explicit Triggers first
    for trigger in triggers:
        trigger_index = lower_input.find(trigger)
        if trigger_index != -1:
            memory_text = user_input[trigger_index + len(trigger):].strip(" .,:;!?-") # More comprehensive stripping
            if memory_text:
                found_memory_details = {
                    "text": memory_text,
                    "context": f"User explicit request: '{trigger}'",
                    "type": "explicit"
                }
                break

    # 2. If no explicit trigger, check for important keywords
    if not found_memory_details:
         extracted_keywords = sorted(list(set([keyword for keyword in keywords if keyword in lower_input]))) # Unique, sorted

         if extracted_keywords:
             # Store the full user input for now, context is keyword detection
             memory_text = user_input
             found_memory_details = {
                "text": memory_text,
                "context": f"Detected keywords: {', '.join(extracted_keywords)}",
                "type": "inferred_keyword"
             }

    # 3. Add to LTM if relevant memory details were found
    if found_memory_details:
        timestamp = datetime.now().isoformat()
        # Re-extract keywords from the *final* memory text to be saved
        final_keywords = sorted(list(set([kw for kw in keywords if kw in found_memory_details["text"].lower()])))

        new_memory_entry = {
            "timestamp": timestamp,
            "memory_text": found_memory_details["text"],
            "context": found_memory_details["context"],
            "keywords": final_keywords,
            "type": found_memory_details["type"],
            "user_input_context": user_input # Store original input for reference
        }

        # Avoid adding exact duplicates based on 'memory_text' only
        is_duplicate = any(entry.get('memory_text') == new_memory_entry['memory_text'] for entry in long_term_memory)

        if not is_duplicate:
            print(f"\n[LTM] Adding memory: '{new_memory_entry['memory_text']}' (Context: {new_memory_entry['context']})")
            long_term_memory.append(new_memory_entry)
            updated = True
        # else:
            # print("[Debug] Skipping add - duplicate LTM entry based on text.")

    return updated

# Phase 3: Step 7 - Placeholder Personalization Function
# ==============================================================================

def personalize_response(base_response, emotion, user_input):
    """Enhances the base response with personalization."""
    global profile # Ensure access to profile
    personalized_text = base_response
    elder_name = profile.get("name", None) # Use None if name is missing/empty
    added_name = False

    # --- Refined Personalization Logic ---

    # 1. Adjust Tone based on Emotion (Only if base response isn't already similar)
    # Check if base_response already contains common sentiment indicators
    base_lower = personalized_text.lower()
    has_neg_sentiment = any(w in base_lower for w in ["sorry", "difficult", "sad", "unfortunately"])
    has_pos_sentiment = any(w in base_lower for w in ["good", "great", "nice", "wonderful", "happy", "glad"])

    if emotion == 'NEGATIVE' and not has_neg_sentiment:
    # Add checks: don't add if input is very short or a question
        if len(user_input.split()) > 2 and not user_input.strip().endswith("?"):
            empathetic_phrases = ["I'm sorry to hear that.", "That sounds difficult.", "Oh dear, I understand.", "Sending comforting thoughts."]
            personalized_text = random.choice(empathetic_phrases) + " " + personalized_text.strip()
    elif emotion == 'POSITIVE' and not has_pos_sentiment:
         positive_phrases = ["That's wonderful news!", "I'm glad to hear that!", "That sounds lovely!", "Excellent!"]
         # Append less frequently to avoid sounding repetitive
         if random.random() < 0.10:
              personalized_text = personalized_text.strip() + f" {random.choice(positive_phrases)}"

    # 2. Add Name (More context-aware)
    # Add name only if elder_name exists and the response is not too short or generic
    if elder_name and len(personalized_text.split()) > 4 and random.random() < 0.20:
        # Avoid adding if response starts very personally ("I think...", "You should...")
        if not base_lower.startswith(("i ", "you ", "my ", "let's ")) :
             # Prepend if it makes sense grammatically
             personalized_text = f"{elder_name}, {personalized_text[0].lower() + personalized_text[1:]}" # Lowercase first letter after name
             added_name = True
        # elif not base_lower.endswith((".", "!", "?")): # Try appending if prepending is awkward and no punctuation
        #     personalized_text += f", {elder_name}."
        #     added_name = True


    # 3. Reference Interests (If relevant keyword found and not just added name)
    interests = profile.get("interests", [])
    if interests and not added_name:
        matched_interest = None
        for interest in interests:
            # Check for whole word match or common variations if possible
            if interest.lower() in user_input.lower(): # Simple check for now
                # Avoid referencing if the AI response already mentions it
                if interest.lower() not in personalized_text.lower():
                    matched_interest = interest
                    break
        if matched_interest and random.random() < 0.25: # Lower probability
            follow_ups = [
                f"Speaking of {matched_interest}, how is that going?",
                f"That reminds me of your interest in {matched_interest}. Anything new?",
                f"Have you had any time for {matched_interest} lately?"
            ]
            personalized_text += f" {random.choice(follow_ups)}"


    # TODO: LTM recall integration (Needs a separate function to search LTM based on input/context)
    # TODO: Medication reminders (Needs LTM search + time checking)

    return personalized_text.strip()


# Phase 3: Step 8 - Placeholder LTM Startup Prompt
# ==============================================================================

def check_ltm_for_startup_prompt():
    """Checks LTM for relevant items to mention at startup (BASIC VERSION)."""
    global long_term_memory
    # --- Needs more robust implementation ---
    today = datetime.now().date()
    relevant_prompts = []
    # Check last N memories or memories within last X days
    recent_memories = long_term_memory[-20:] # Check last 20 entries for performance

    for memory in reversed(recent_memories): # Check most recent first
         try:
             mem_time = datetime.fromisoformat(memory['timestamp']).date()
             days_diff = (today - mem_time).days

             # Look for future events or recent past events
             if days_diff >= -3 and days_diff < 7: # Look for events coming up in 3 days or happened in last week
                 text_lower = memory['memory_text'].lower()
                 prompt = None

                 # Prioritize future events
                 if days_diff >= -3 and days_diff <= 0 : # Today or next 3 days
                      if "appointment" in text_lower or "doctor" in text_lower or "visit" in text_lower:
                           prompt = f"I remember you mentioned an appointment or visit soon ('{memory['memory_text']}'). Is that coming up?"
                      elif "birthday" in text_lower:
                           prompt = f"Wasn't there a birthday coming up? I recall you mentioning '{memory['memory_text']}'."
                      elif "travel" in text_lower or "plan" in text_lower:
                           prompt = f"I remember you were planning something related to '{memory['memory_text']}'. How are those plans going?"

                 # Check recent past events only if no future prompt found yet
                 elif not relevant_prompts and days_diff > 0: # Happened in last week
                      if "birthday" in text_lower or "anniversary" in text_lower or "wedding" in text_lower:
                           prompt = f"I remember you mentioned '{memory['memory_text']}' recently. How did that go?"
                      elif "visit" in text_lower and not ("doctor" in text_lower or "appointment" in text_lower): # Social visit
                           prompt = f"You mentioned a visit recently ('{memory['memory_text']}'). How was it?"

                 # Add unique prompts
                 if prompt and prompt not in relevant_prompts:
                      relevant_prompts.append(prompt)
                      # Break after finding one good prompt to avoid overwhelming user
                      break

         except (ValueError, KeyError, TypeError, IndexError) as e: # Added IndexError
             # print(f"[Debug] Error processing LTM entry for startup prompt: {e}, Entry: {memory}")
             continue

    if relevant_prompts:
        return relevant_prompts[0] # Return the first (most relevant/recent) prompt found
    return None


# Main Application Logic
# ==============================================================================

def main():
    """Runs the main conversation loop."""
    global profile, long_term_memory, short_term_memory # Ensure globals are accessible

    # Initial greeting using loaded profile name
    print("\n" + "="*30)
    print(" Conversational AI Companion")
    print("="*30)
    # Use profile directly now as it's loaded before main() is called
    print(f"AI: Hello {profile.get('name', 'there')}! I'm ready to chat. Type 'quit' or 'bye' to exit.")

    # Check for startup prompt from LTM
    startup_prompt = check_ltm_for_startup_prompt()
    if startup_prompt:
        print(f"AI: {startup_prompt}")
        # Add AI's startup prompt to STM for context, user input is empty
        update_short_term_memory("", startup_prompt)

    # --- Conversation Loop ---
    while True:
        try:
            # Use profile name in prompt
            user_input = input(f"{profile.get('name', 'User')}: ")
        except EOFError:
             print("\nAI: Input stream closed. Goodbye!")
             break
        except KeyboardInterrupt: # Catch Ctrl+C here too for graceful exit
             print("\nAI: Okay, signing off now. Goodbye!")
             break

        if not user_input.strip():
            continue

        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye', 'stop']:
            break

        # 1. Detect Emotion
        emotion = detect_emotion(user_input)
        # <<< CHANGE 4: Add Emotion Debug Print >>>
        print(f"--> [DEBUG] Detected Emotion: {emotion}")

        # 2. Generate Base Response
        base_response = generate_blenderbot_response(user_input, short_term_memory)
        # <<< CHANGE 5: Add Base Response Debug Print >>>
        print(f"--> [DEBUG] Base Response from Model: {base_response}")

        # 3. Personalize Response
        final_response = personalize_response(base_response, emotion, user_input)

        # 4. Output Final Response
        print(f"AI: {final_response}")

        # 5. Update Memories
        update_short_term_memory(user_input, final_response)

        # Check user input for LTM entries and save if updated
        ltm_updated = check_and_update_long_term_memory(user_input)
        if ltm_updated:
            save_ltm(LTM_FILE, long_term_memory)

    # --- End of Loop ---
    print(f"\nAI: It was lovely chatting with you, {profile.get('name', 'take care')}! Goodbye!")
    # Final save of LTM before exiting (good practice)
    save_ltm(LTM_FILE, long_term_memory)


# Script Entry Point - Load models here before starting main loop
# ==============================================================================
if __name__ == "__main__":
    print("Initializing AI Companion...")

    # --- Load Profile ---
    profile = load_profile(PROFILE_FILE)
    if not profile:
        # Provide a default profile if loading fails, allowing continuation
        print("Warning: Could not load profile or profile is empty. Using default values.")
        profile = {"name": "User", "interests": [], "family": {}}
        # exit() # Decide if you want to exit or continue with defaults
    print(f"Profile loaded for: {profile.get('name', 'Unknown User')}")

    # --- Load LTM ---
    # Ensure LTM file path uses profile info if needed, or is fixed
    # For simplicity, keeping LTM_FILE fixed for now.
    long_term_memory = load_ltm(LTM_FILE)
    print(f"Loaded {len(long_term_memory)} entries from Long-Term Memory.")

    # --- Determine Device (GPU/CPU) ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if str(device) == "cuda":
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error detecting device: {e}. Defaulting to CPU.")
        device = torch.device("cpu")


    # --- Load Emotion Detection Model ---
    print("Loading emotion detection model...")
    try:
        emotion_classifier = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            device=0 if str(device) == "cuda" else -1 # Use device index for pipeline
            )
        print("Emotion detection model loaded successfully.")
    except Exception as e:
        print(f"Warning: Error loading emotion detection model: {e}")
        print("Continuing without emotion detection.")
        emotion_classifier = None # Ensure it's None if loading failed

    # --- Load Conversational Model ---
    fine_tuned_model_path = r"F:\PROJECT\elder_ai_companion\fine_tuned_model1"
    base_model_name = 'facebook/blenderbot-400M-distill'

    if not os.path.isdir(fine_tuned_model_path):
        print(f"CRITICAL Error: The specified fine-tuned model path does not exist:")
        print(f"'{fine_tuned_model_path}'")
        print("Please ensure the path is correct.")
        exit()

    print(f"Loading fine-tuned conversational model weights from: {fine_tuned_model_path}")
    print(f"Loading tokenizer from original base model: {base_model_name}")

    try:
        print(f"Attempting to load tokenizer '{base_model_name}' from Hugging Face Hub...")
        conversational_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("Tokenizer loaded successfully.")

        print(f"Attempting to load model weights (using BlenderbotForConditionalGeneration) from local path: {fine_tuned_model_path}")
        # Load model using the determined device directly if possible
        # Note: `from_pretrained` doesn't always accept device; move after loading
        conversational_model = BlenderbotForConditionalGeneration.from_pretrained(fine_tuned_model_path)
        print("Model weights loaded successfully from local path.")

        # <<< CHANGE 6: Add Model Config Debug Print >>>
        print(f"--> [DEBUG] Loaded model config - max_position_embeddings: {getattr(conversational_model.config, 'max_position_embeddings', 'Not Found')}")
        print(f"--> [DEBUG] Loaded model config - vocab_size: {getattr(conversational_model.config, 'vocab_size', 'Not Found')}")
        # <<< END DEBUG PRINT >>>

        conversational_model.to(device) # Move model to GPU or CPU
        conversational_model.eval() # Set model to evaluation mode (important!)
        print(f"Fine-tuned conversational model moved to device '{device}' and set to evaluation mode.")

    except OSError as e:
         print(f"CRITICAL Error loading model/tokenizer: {e}")
         traceback.print_exc() # Show full traceback for OSError
         if "config.json" in str(e) and fine_tuned_model_path in str(e):
              print(f"\n*** Missing 'config.json' in your local directory: {fine_tuned_model_path}. ***")
         elif (".safetensors" in str(e) or ".bin" in str(e)) and fine_tuned_model_path in str(e):
              print(f"\n*** Missing model weights file (e.g., 'model.safetensors') in {fine_tuned_model_path}. ***")
         elif base_model_name in str(e):
              print(f"\n*** Error loading tokenizer '{base_model_name}'. Check internet connection/model name. ***")
         print("Exiting application.")
         exit()
    except ImportError as e:
         print(f"CRITICAL Error: Missing required library: {e}")
         traceback.print_exc()
         # Add suggestions based on common libraries
         if "sentencepiece" in str(e).lower(): print("Try: pip install sentencepiece")
         if "protobuf" in str(e).lower(): print("Try: pip install protobuf")
         if "tokenizers" in str(e).lower(): print("Try: pip install tokenizers")
         print("Exiting application.")
         exit()
    except Exception as e:
        print(f"CRITICAL Error during model loading: {e}")
        traceback.print_exc() # Print full traceback for any other loading error
        if "out of memory" in str(e).lower():
            print("\nCUDA Out of Memory Error. Your GPU may not have enough RAM for this model.")
            print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0) if str(device)=='cuda' else 'N/A'}")
            print("Try running on CPU (if not already) or use a smaller model.")
        print("Exiting application.")
        exit()

    # --- Start the main application ---
    try:
        main()
    # Moved KeyboardInterrupt handling inside main's loop for cleaner exit
    except Exception as e:
        # Catch any unexpected errors during main execution
        print(f"\n--- An unexpected error occurred during the main loop! ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        # Attempt final save even on error
        print("\nAttempting final save of LTM...")
        save_ltm(LTM_FILE, long_term_memory)

    print("\nApplication finished.")
# --------------------------------------