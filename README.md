# Meeting_summarizer

Overview
The Meeting Summarizer is a Python-based tool that:
Converts speech to text from an audio file using Whisper AI.
Detects the language of the transcript.
Translates non-English transcripts to English using MBart.
Summarizes the meeting using BART (Facebook's bart-large-cnn model).
Extracts and assigns action items/tasks from the meeting.
Saves the results to a text file.

Output
The script will:
Transcribe the meeting audio.
Detect the language of the transcript.
Translate it to English if necessary.
Summarize the key points.
Extract and assign tasks to participants.
Save results in meeting_summary.txt.
