import os
import librosa
import whisper
from transformers import (
    pipeline,
    MBartForConditionalGeneration, 
    MBartTokenizer,
    BartForConditionalGeneration, 
    BartTokenizer
)
from langdetect import detect
import re

print("Starting Meeting Summarizer...")

class MeetingSummarizer:
    def _init_(self):
        print("Loading models... (this may take a few minutes on first run)")
        
        # Load audio transcription model (Whisper for better accuracy)
        self.whisper_model = whisper.load_model("large")
        
        # Load summarization model (alternative model for better results)
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn"
        )
        
        # Load translation model for non-English transcripts
        self.translator_tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
        self.translator_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
        
        print("All models loaded successfully!")
    
    def load_audio(self, file_path):
        """Load audio file for processing"""
        print(f"Loading audio file: {file_path}")
        try:
            # Load audio using librosa (handles mp3 format)
            audio, sample_rate = librosa.load(file_path, sr=16000)
            print(f"Audio loaded: {len(audio)/sample_rate:.2f} seconds")
            return audio, sample_rate
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def transcribe(self, audio_file):
        """Convert speech to text using Whisper"""
        print("Transcribing audio using Whisper...")
        try:
            result = self.whisper_model.transcribe(audio_file)
            text = result["text"]
            print(f"Transcription complete: {len(text)} characters")
            return text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return "Transcription failed. Please check the audio file."
    
    def detect_language(self, text):
        """Detect the language of the transcribed text"""
        print("Detecting language...")
        try:
            lang_code = detect(text)
            
            # Map language codes to full names
            language_names = {
                'en': 'English',
                'fr': 'French',
                'es': 'Spanish',
                'de': 'German',
                'it': 'Italian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese',
                'ru': 'Russian',
                'pt': 'Portuguese',
                'nl': 'Dutch',
                'ar': 'Arabic',
                'hi': 'Hindi'
            }
            
            lang_name = language_names.get(lang_code, f"Unknown ({lang_code})")
            print(f"Detected language: {lang_name}")
            return lang_code, lang_name
        except Exception as e:
            print(f"Error detecting language: {e}")
            return "unknown", "Unknown"
    
    def translate_if_needed(self, text, source_lang):
        """Translate text to English if not already in English"""
        if source_lang == 'en':
            print("Text already in English, no translation needed")
            return text
        
        print(f"Translating from {source_lang} to English...")
        try:
            # Set up translation
            self.translator_tokenizer.src_lang = f"{source_lang}_XX"
            
            # Encode text for translation
            encoded = self.translator_tokenizer(text, return_tensors="pt")
            
            # Generate translation
            translated_tokens = self.translator_model.generate(
                **encoded,
                forced_bos_token_id=self.translator_tokenizer.lang_code_to_id["en_XX"],
                max_length=1024
            )
            
            # Decode translation
            translation = self.translator_tokenizer.batch_decode(
                translated_tokens, 
                skip_special_tokens=True
            )[0]
            
            print("Translation complete")
            return translation
        except Exception as e:
            print(f"Error during translation: {e}")
            return text  # Return original text if translation fails
    
    def summarize(self, text):
        """Generate a summary of the meeting transcript"""
        print("Generating summary...")
        try:
            # Use a direct approach with specific parameters
            from transformers import BartForConditionalGeneration, BartTokenizer
            
            print("Loading summarization model...")
            model_name = "facebook/bart-large-cnn"
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
            
            # Process text in chunks for long transcripts
            max_length = 1024  # BART's maximum input length
            chunks = []
            
            # Split text into chunks that fit into model
            print("Processing text for summarization...")
            if len(text) > max_length * 3:  # If text is very long
                # Split by sentences to maintain context
                sentences = text.split('.')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_length:
                        current_chunk += sentence + "."
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + "."
                
                if current_chunk:  # Add the last chunk
                    chunks.append(current_chunk)
                    
                print(f"Split text into {len(chunks)} chunks for processing")
            else:
                chunks = [text]
                
            # Process each chunk
            summaries = []
            for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks for reasonable processing time
                print(f"Summarizing chunk {i+1} of {min(len(chunks), 5)}...")
                
                # Encode text
                inputs = tokenizer([chunk], max_length=1024, return_tensors="pt", truncation=True)
                
                # Generate summary
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    num_beams=4,
                    min_length=100,  # Increased minimum summary length
                    max_length=300,  # Increased maximum summary length
                    early_stopping=True
                )
                
                # Decode summary
                chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(chunk_summary)
                print(f"Chunk {i+1} summarization complete")
            
            # Combine chunk summaries
            full_summary = " ".join(summaries)
            
            # If we have multiple chunks, summarize the combined summary again
            if len(summaries) > 1:
                print("Creating final summary from chunk summaries...")
                inputs = tokenizer([full_summary], max_length=1024, return_tensors="pt", truncation=True)
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    min_length=100,
                    max_length=300,
                    early_stopping=True
                )
                full_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            print("Summary generation completed successfully")
            return full_summary
            
        except Exception as e:
            import traceback
            print(f"Error during summarization: {e}")
            print(traceback.format_exc())  # Print full error details
            
            # Create a basic fallback summary
            print("Creating basic fallback summary...")
            try:
                sentences = text.split('.')
                important_sentences = []
                
                # Try to extract important sentences containing key words
                keywords = ["important", "decision", "agreed", "plan", "deadline", "task", "action", "conclude"]
                
                for sentence in sentences[:50]:  # Check first 50 sentences
                    if any(keyword in sentence.lower() for keyword in keywords):
                        important_sentences.append(sentence)
                
                # If we found important sentences, use them
                if len(important_sentences) >= 3:
                    return ". ".join(important_sentences[:5]) + "."
                # Otherwise use the first few sentences
                elif len(sentences) > 5:
                    return ". ".join(sentences[:5]) + "."
                else:
                    return text
                    
            except:
                # Absolute fallback
                return "Summary generation failed. Please check the transcript manually."
    
    def extract_tasks(self, text):
        """Extract tasks mentioned in the meeting"""
        print("Extracting tasks...")
        try:
            # Look for common task patterns
            task_patterns = [
                r"(?:need to|needs to|needed to) ([^\.;!?]*)",
                r"(?:should|shall|must|will|going to) ([^\.;!?]*)",
                r"(?:have to|has to|had to) ([^\.;!?]*)",
                r"(?:assigned to|responsible for) ([^\.;!?]*)",
                r"action item[s]?:? ([^\.;!?]*)",
                r"task[s]?:? ([^\.;!?]*)",
                r"(?:please|kindly) ([^\.;!?]*)",  # Added more patterns
                r"(?:next step[s]?:?) ([^\.;!?]*)"
            ]
            
            # Find matching tasks
            tasks = []
            for pattern in task_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    task = match.group(1).strip()
                    if task and len(task) > 3 and task not in tasks:
                        tasks.append(task)
            
            # If no tasks found, look for sentences with action verbs
            if not tasks:
                action_verbs = ["create", "develop", "implement", "design", "research", 
                               "prepare", "review", "schedule", "organize", "contact", 
                               "follow up", "complete", "finish", "submit"]
                
                sentences = re.split(r'[.!?]', text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if any(verb in sentence.lower() for verb in action_verbs):
                        if sentence and sentence not in tasks:
                            tasks.append(sentence)
            
            print(f"Found {len(tasks)} potential tasks")
            return tasks
        except Exception as e:
            print(f"Error extracting tasks: {e}")
            return ["Task extraction failed"]
    
    def assign_tasks(self, tasks, text):
        """Assign tasks to meeting participants"""
        print("Assigning tasks to people...")
        try:
            # Extract potential names (simple approach)
            # Look for capitalized words that might be names
            name_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)'
            potential_names = set(re.findall(name_pattern, text))
            
            # Filter out common false positives
            common_terms = {"Next Week", "This Month", "Last Year", "Thank You"}
            potential_names = [name for name in potential_names if name not in common_terms]
            
            # If no names found, use generic assignments
            if not potential_names:
                potential_names = ["Team Member", "Project Owner", "Participant"]
            
            # Assign tasks in round-robin fashion
            assigned_tasks = []
            for i, task in enumerate(tasks):
                person = potential_names[i % len(potential_names)]
                assigned_tasks.append(f"{person}: {task}")
            
            print(f"Assigned {len(assigned_tasks)} tasks")
            return assigned_tasks
        except Exception as e:
            print(f"Error assigning tasks: {e}")
            return tasks  # Return unassigned tasks if assignment fails
    
    def process_meeting(self, audio_file):
        """Process a meeting audio file end-to-end"""
        print(f"\n===== PROCESSING MEETING: {audio_file} =====\n")
        
        # Step 1: Verify the audio file path
        print(f"Audio file path: {os.path.abspath(audio_file)}")
        if not os.path.exists(audio_file):
            return {"error": f"File '{audio_file}' not found. Make sure it's in the correct folder."}
        
        # Step 2: Transcribe audio to text
        transcription = self.transcribe(audio_file)
        print("\n--- TRANSCRIPTION ---")
        print(transcription)
        
        # Step 3: Detect language of transcription
        lang_code, lang_name = self.detect_language(transcription)
        print("\n--- LANGUAGE ---")
        print(f"Detected: {lang_name} ({lang_code})")
        
        # Step 4: Translate if not in English
        if lang_code != "en":
            text_for_processing = self.translate_if_needed(transcription, lang_code)
            translation = text_for_processing
            print("\n--- TRANSLATION ---")
            print(translation)
        else:
            text_for_processing = transcription
            translation = None
        
        # Step 5: Generate summary
        summary = self.summarize(text_for_processing)
        print("\n--- SUMMARY ---")
        print(summary)
        
        # Step 6: Extract tasks
        tasks = self.extract_tasks(text_for_processing)
        print("\n--- TASKS ---")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
        
        # Step 7: Assign tasks to people
        assigned_tasks = self.assign_tasks(tasks, text_for_processing)
        print("\n--- ASSIGNED TASKS ---")
        for i, task in enumerate(assigned_tasks, 1):
            print(f"{i}. {task}")
        
        # Create results object
        results = {
            "transcription": transcription,
            "language": {
                "code": lang_code,
                "name": lang_name
            },
            "translation": translation,
            "summary": summary,
            "tasks": assigned_tasks
        }
        
        return results
    
    def display_results(self, results):
        """Show results in the console"""
        print("\n" + "="*60)
        print(" "*20 + "MEETING SUMMARY")
        print("="*60)
        
        print("\n--- TRANSCRIPTION ---")
        if len(results["transcription"]) > 300:
            print(results["transcription"][:300] + "...")
            print(f"(Full transcription: {len(results['transcription'])} characters)")
        else:
            print(results["transcription"])
        
        print("\n--- LANGUAGE ---")
        print(f"Detected: {results['language']['name']} ({results['language']['code']})")
        
        if results.get("translation"):
            print("\n--- TRANSLATION ---")
            if len(results["translation"]) > 300:
                print(results["translation"][:300] + "...")
            else:
                print(results["translation"])
        
        print("\n--- SUMMARY ---")
        print(results["summary"])
        
        print("\n--- ASSIGNED TASKS ---")
        for i, task in enumerate(results["tasks"], 1):
            print(f"• {task}")  # Use bullet points
        
        print("\n" + "="*60)
    
    def save_results(self, results, output_file="meeting_summary.txt"):
        """Save results to a text file"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("="*60 + "\n")
                f.write(" "*20 + "MEETING SUMMARY\n")
                f.write("="*60 + "\n\n")
                
                f.write("--- TRANSCRIPTION ---\n")
                f.write(results["transcription"] + "\n\n")
                
                f.write("--- LANGUAGE ---\n")
                f.write(f"Detected: {results['language']['name']} ({results['language']['code']})\n\n")
                
                if results.get("translation"):
                    f.write("--- TRANSLATION ---\n")
                    f.write(results["translation"] + "\n\n")
                
                f.write("--- SUMMARY ---\n")
                f.write(results["summary"] + "\n\n")
                
                f.write("--- ASSIGNED TASKS ---\n")
                for i, task in enumerate(results["tasks"], 1):
                    f.write(f"• {task}\n")  # Use bullet points
                
                f.write("\n" + "="*60 + "\n")
            
            print(f"\nResults saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

# Main execution
def main():
    # File to process
    audio_file = "meeting.mp3"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found. Make sure it's in the same folder as this script.")
        return
    
    # Initialize the summarizer
    summarizer = MeetingSummarizer()
    
    # Process the meeting
    results = summarizer.process_meeting(audio_file)
    
    # Display and save results
    if "error" not in results:
        summarizer.display_results(results)
        summarizer.save_results(results)
    else:
        print(f"Error: {results['error']}")

if _name_ == "_main_":
    main()
