import streamlit as st
import numpy as np
import random
import os
import json
import pickle
# import tensorflow as tf  # Temporarily commented out for deployment testing
from music21 import stream, note, tempo, meter, key
import tempfile
import uuid
import wave # Used for WAV file creation

# --- Configuration Paths ---
MODEL_PATH = 'models/ai_music_model.keras'
MAPPINGS_PATH = 'models/note_mappings.pkl'
METADATA_PATH = 'models/model_metadata.json'

# --- Deployment Test ---
# st.success("🚀 **Deployment Successful!** - App is running correctly.")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Music Generator",
    layout="centered", # Matches a typical Flask page layout
    initial_sidebar_state="collapsed"
)

# --- Custom Styling (Mimicking original UI aesthetics with Streamlit's capabilities) ---
st.markdown("""
<style>
    /* Overall App Background & Font */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Poppins', sans-serif; /* Using a common web font */
        color: #f0f0f0; /* Light text color */
    }

    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Mimic Navbar */
    .navbar-streamlit {
        background-color: #333;
        padding: 10px 0;
        text-align: center;
        margin-bottom: 20px; /* Space below navbar */
        border-radius: 10px; /* Rounded corners for navbar */
    }
    .navbar-streamlit a {
        color: white;
        margin: 0 15px;
        text-decoration: none;
        font-weight: bold;
        transition: color 0.3s ease;
    }
    .navbar-streamlit a:hover {
        color: #add8e6; /* Light blue on hover */
    }

    /* Mimic Card Style */
    .stContainer {
        background-color: rgba(255, 255, 255, 0.1); /* Semi-transparent white */
        border-radius: 15px; /* Rounded corners */
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        backdrop-filter: blur(5px); /* Frosted glass effect */
        -webkit-backdrop-filter: blur(5px); /* For Safari */
        border: 1px solid rgba(255, 255, 255, 0.2); /* Light border */
    }

    /* Card Header */
    .card-header-primary {
        background-color: #007bff; /* Primary blue */
        color: white;
        padding: 15px 20px;
        border-top-left-radius: 15px;
        border-top-right-radius: 15px;
        margin: -20px -20px 20px -20px; /* Adjust margin to fit card */
        font-size: 1.5em;
        font-weight: bold;
    }
    .card-header-info {
        background-color: #17a2b8; /* Info blue */
        color: white;
        padding: 15px 20px;
        border-top-left-radius: 15px;
        border-top-right-radius: 15px;
        margin: -20px -20px 20px -20px;
        font-size: 1.5em;
        font-weight: bold;
    }
    .card-header-success {
        background-color: #28a745; /* Success green */
        color: white;
        padding: 15px 20px;
        border-top-left-radius: 15px;
        border-top-right-radius: 15px;
        margin: -20px -20px 20px -20px;
        font-size: 1.5em;
        font-weight: bold;
    }


    /* Headings within cards */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff; /* White headings */
        margin-top: 0;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #007bff; /* Primary blue */
        color: white;
        border-radius: 8px; /* More rounded */
        border: none;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        transform: translateY(-2px); /* Slight lift effect */
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #add8e6; /* Light blue track */
    }
    .stSlider > div > div > div > div > div {
        background-color: #007bff; /* Primary blue thumb */
    }

    /* Text Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: #f0f0f0;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 8px 12px;
    }
    .stTextInput > label {
        color: #ffffff; /* Label color */
        font-weight: bold;
    }

    /* Metrics (for model info) */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stMetric > div:first-child { /* Label */
        color: #cccccc;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .stMetric > div:nth-child(2) { /* Value */
        font-size: 1.5em;
        font-weight: bold;
        color: #ffffff;
    }
    /* Specific colors for metrics */
    .stMetric .text-success { color: #28a745 !important; }
    .stMetric .text-info { color: #17a2b8 !important; }
    .stMetric .text-warning { color: #ffc107 !important; }
    .stMetric .text-primary { color: #007bff !important; }


    /* Alerts/Messages */
    .stAlert {
        border-radius: 10px;
    }

    /* Footer */
    .footer-streamlit {
        text-align: center;
        padding: 20px;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 50px;
        font-size: 0.9em;
    }

</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if 'generated_notes' not in st.session_state:
    st.session_state.generated_notes = None
if 'midi_data' not in st.session_state:
    st.session_state.midi_data = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'midi_filename' not in st.session_state:
    st.session_state.midi_filename = None
if 'audio_filename' not in st.session_state:
    st.session_state.audio_filename = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}
if 'show_about' not in st.session_state:
    st.session_state.show_about = False


# --- Model Loading (Cached for efficiency) ---
@st.cache_resource
def load_ai_model_cached():
    """Load the AI music model and mappings, cached by Streamlit."""
    try:
        # Try to load TensorFlow and model if available
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
        except ImportError:
            model = None
        except Exception as e:
            model = None
        
        # Load mappings and metadata
        with open(MAPPINGS_PATH, 'rb') as f:
            note_mappings = pickle.load(f)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        return model, note_mappings, metadata
    except Exception as e:
        return None, None, None

# --- Core Music Generation Functions (Reused) ---

def generate_music_sequence(num_notes, temperature, seed, model, note_mappings):
    """Generate music using the AI model"""
    if model is None:
        # Generate simple random notes as fallback
        notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        return [random.choice(notes) for _ in range(num_notes)]
    
    if note_mappings is None:
        st.error("Note mappings not loaded. Cannot generate music.")
        return None
    
    int_to_note = note_mappings.get('int_to_note')
    if not int_to_note:
        st.error("Note mappings (int_to_note) not found in loaded data.")
        return None

    sequence_length = note_mappings.get('sequence_length')
    if sequence_length is None:
        st.error("Sequence length not found in loaded data.")
        return None

    vocab_size = len(int_to_note)
    
    if vocab_size == 0:
        st.error("Empty vocabulary. Cannot generate music.")
        return None

    # Set random seed for reproducibility if provided
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    current_sequence = [random.randint(0, vocab_size - 1) for _ in range(sequence_length)]
    generated_notes = []
    
    try:
        if model is None:
            # Fallback: generate simple random notes
            notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
            return [random.choice(notes) for _ in range(num_notes)]
        
        # AI model generation
        for i in range(num_notes):
            input_sequence = np.array([current_sequence], dtype=np.int32)
            
            prediction = model.predict(input_sequence, verbose=0)[0]
            
            if temperature != 1.0:
                prediction = np.log(np.clip(prediction, 1e-7, 1.0)) / temperature
                exp_preds = np.exp(prediction)
                prediction = exp_preds / np.sum(exp_preds)
            
            prediction = np.clip(prediction, 1e-7, 1.0)
            prediction = prediction / np.sum(prediction)
            
            next_note_idx = np.random.choice(len(prediction), p=prediction)
            
            try:
                next_note_idx = int(next_note_idx) 
                
                if next_note_idx in int_to_note:
                    next_note = str(int_to_note[next_note_idx])
                elif str(next_note_idx) in int_to_note:
                    next_note = str(int_to_note[str(next_note_idx)])
                else:
                    st.warning(f"Note index {next_note_idx} not found in mappings. Falling back to C4.")
                    next_note = 'C4'
                    note_to_int = note_mappings.get('note_to_int', {})
                    next_note_idx = note_to_int.get('C4', 0)
                
            except (KeyError, TypeError, ValueError) as e:
                st.warning(f"Error converting note index: {e}. Falling back to C4.")
                next_note = 'C4'
                note_to_int = note_mappings.get('note_to_int', {})
                next_note_idx = note_to_int.get('C4', 0)
            
            generated_notes.append(next_note)
            current_sequence = current_sequence[1:] + [int(next_note_idx)]
            
    except Exception as e:
        st.error(f"Error during music generation: {e}")
        return None
    
    return generated_notes

def notes_to_midi(notes, output_path, tempo_bpm):
    """Convert notes to MIDI file"""
    try:
        composition = stream.Stream()
        composition.append(tempo.TempoIndication(number=tempo_bpm))
        composition.append(meter.TimeSignature('4/4'))
        composition.append(key.KeySignature(0))
        
        for note_name in notes:
            try:
                if note_name and note_name != 'REST' and note_name.strip():
                    clean_note = str(note_name).strip()
                    if clean_note:
                        new_note = note.Note(clean_note)
                        new_note.quarterLength = 0.5 # Standard quarter note duration
                        composition.append(new_note)
            except Exception as note_error:
                st.warning(f"Skipping problematic note: {note_name} due to error during MIDI conversion: {note_error}")
                continue
        
        composition.write('midi', fp=output_path)
        return output_path
    except Exception as e:
        st.error(f"Error creating MIDI: {e}")
        return None

def create_browser_compatible_audio(notes, output_path, tempo_bpm):
    """Create a simple WAV audio representation for browser playback"""
    try:
        note_frequencies = {
            'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
            'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
            'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
            'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46,
            'G5': 783.99, 'A5': 880.00, 'B5': 987.77, 'C3': 130.81,
            'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00,
            'A3': 220.00, 'B3': 246.94
        }
        
        sample_rate = 22050
        note_duration = 60.0 / tempo_bpm
        
        audio_data = []
        for i, note_name in enumerate(notes[:60]): # Limit to prevent long processing for browser audio
            if note_name in note_frequencies:
                freq = note_frequencies[note_name]
                t = np.linspace(0, note_duration, int(sample_rate * note_duration), endpoint=False)
                
                envelope = np.ones_like(t)
                attack_samples = int(0.1 * len(t))
                release_samples = int(0.1 * len(t))
                
                if attack_samples > 0:
                    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                if release_samples > 0:
                    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
                
                wave_data = np.sin(2 * np.pi * freq * t) * envelope * 0.3 # Reduce amplitude
                audio_data.extend(wave_data)
            else:
                silence = np.zeros(int(sample_rate * note_duration))
                audio_data.extend(silence)
        
        if audio_data:
            audio_array = np.array(audio_data, dtype=np.float32)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = np.int16(audio_array * 32767)
            
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            return True
        return False
        
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return False

# --- Streamlit UI Layout ---

# Mimic Navbar
st.markdown("""
<div class="navbar-streamlit">
    <a href="#home">AI Music Generator</a>
    <a href="#about-the-ai-music-generator">About</a>
</div>
""", unsafe_allow_html=True)


# Main title mimicking h1 from base.html
st.markdown('<h1 id="home" style="text-align: center;">AI Music Generator</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: rgba(255, 255, 255, 0.8);">Create original music compositions using artificial intelligence</p>', unsafe_allow_html=True)

# Main content container
with st.container(border=False): # Use container to mimic card
    st.markdown('<div class="card-header-primary">🤖 AI Music Generator</div>', unsafe_allow_html=True)
    
    # Load model and mappings at the start of the script
    model_loaded, note_mappings_loaded, metadata_loaded = load_ai_model_cached()

    if model_loaded is None and note_mappings_loaded is None:
        st.error("❌ Application not ready: Model files could not be loaded. Please check logs for details.")
    else:
        with st.form("music_generation_form"):
            col1, col2 = st.columns(2) # Two columns for layout like your Flask UI
            with col1:
                st.markdown("🎵 **Number of Notes**", unsafe_allow_html=True)
                num_notes = st.slider("", min_value=20, max_value=200, value=st.session_state.last_params.get("num_notes", 80), key="s_num_notes")
                st.markdown(f'<small style="color: rgba(255, 255, 255, 0.7);">Notes: {num_notes}</small>', unsafe_allow_html=True)

                st.markdown("🎶 **Tempo (BPM)**", unsafe_allow_html=True)
                tempo = st.slider("", min_value=60, max_value=200, value=st.session_state.last_params.get("tempo", 120), key="s_tempo")
                st.markdown(f'<small style="color: rgba(255, 255, 255, 0.7);">BPM: {tempo}</small>', unsafe_allow_html=True)

            with col2:
                st.markdown("🎨 **Creativity Level**", unsafe_allow_html=True)
                temperature_options = {
                    "Very Conservative (0.5)": 0.5,
                    "Conservative (0.8)": 0.8,
                    "Balanced (1.0)": 1.0,
                    "Creative (1.2)": 1.2,
                    "Highly Experimental (1.5)": 1.5
                }
                # Find the key for the current value to set default in selectbox
                default_temp_key = next((k for k, v in temperature_options.items() if v == st.session_state.last_params.get("temperature", 1.0)), "Balanced (1.0)")
                
                selected_temp_label = st.selectbox("", options=list(temperature_options.keys()), index=list(temperature_options.keys()).index(default_temp_key), key="s_temperature_select")
                temperature = temperature_options[selected_temp_label]
                
                st.markdown("🌱 **Seed (Optional)**", unsafe_allow_html=True)
                seed_input = st.text_input("Leave empty for random", value=st.session_state.last_params.get("seed_input", ""), key="s_seed", help="Enter an integer to get the same music sequence each time.")

            st.markdown("---") # Separator within form
            submit_button = st.form_submit_button("Generate AI Music 🎵")

        if submit_button:
            seed = None
            if seed_input and str(seed_input).strip():
                try:
                    seed = int(seed_input)
                except ValueError:
                    st.warning("Invalid seed. Please enter an integer or leave blank.")
            
            with st.spinner("Generating music... This might take a moment. Please wait."):
                generated_notes = generate_music_sequence(num_notes, temperature, seed, model_loaded, note_mappings_loaded)

            if generated_notes is None or len(generated_notes) == 0:
                st.error("Failed to generate music - no notes produced.")
                st.session_state.generated_notes = None # Clear state if generation fails
            else:
                st.success("Music generated successfully! Check below for playback and download options.")
                
                temp_dir = tempfile.mkdtemp()
                midi_filename = f'generated_music_{uuid.uuid4().hex[:8]}.mid'
                audio_filename = f'generated_music_{uuid.uuid4().hex[:8]}.wav'
                
                midi_path = os.path.join(temp_dir, midi_filename)
                audio_path = os.path.join(temp_dir, audio_filename)
                
                # Generate MIDI and WAV
                midi_result = notes_to_midi(generated_notes, midi_path, tempo)
                audio_success = create_browser_compatible_audio(generated_notes, audio_path, tempo)

                # Store generated data and metadata in session state as bytes
                st.session_state.generated_notes = generated_notes
                st.session_state.last_params = {
                    "num_notes": num_notes,
                    "temperature": temperature,
                    "tempo": tempo,
                    "seed_input": seed_input
                }
                st.session_state.midi_filename = midi_filename
                st.session_state.audio_filename = audio_filename
                
                st.session_state.midi_data = None
                if midi_result and os.path.exists(midi_path):
                    with open(midi_path, "rb") as f:
                        st.session_state.midi_data = f.read()
                else:
                    st.error("Failed to read generated MIDI file into session state.")

                st.session_state.audio_data = None
                if audio_success and os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        st.session_state.audio_data = f.read()
                else:
                    st.error("Failed to read generated WAV file into session state.")

                # Clean up temporary files immediately after reading into session state
                try:
                    if os.path.exists(midi_path): os.remove(midi_path)
                    if os.path.exists(audio_path): os.remove(audio_path)
                    if os.path.exists(temp_dir): os.rmdir(temp_dir)
                except Exception as e:
                    st.error(f"Error cleaning up temporary files: {e}")
                
                st.rerun() # Force a rerun to display results from session state

    # This block now always runs if generated_notes exists in session state
    if st.session_state.generated_notes is not None:
        st.markdown('<div class="card-header-success">✅ Your AI Composition</div>', unsafe_allow_html=True)
        
        # Two columns for layout: audio player/details and download/share buttons
        col_comp_details, col_comp_actions = st.columns([2, 1])

        with col_comp_details:
            if st.session_state.audio_data:
                st.audio(st.session_state.audio_data, format="audio/wav", start_time=0)
            else:
                st.warning("No WAV audio available for playback.")

            st.markdown("🎵 **Composition Details:**", unsafe_allow_html=True)
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Notes: {len(st.session_state.generated_notes)}")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Creativity: {st.session_state.last_params.get('temperature', 'N/A')}")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Tempo: {st.session_state.last_params.get('tempo', 'N/A')} BPM")
            
            st.markdown("📝 **Note Preview:**", unsafe_allow_html=True)
            st.markdown(f"`{', '.join(st.session_state.generated_notes[:10])} ...`")

        with col_comp_actions:
            if st.session_state.midi_data:
                st.download_button(
                    label="⬇️ Download MIDI",
                    data=st.session_state.midi_data,
                    file_name=st.session_state.midi_filename,
                    mime="audio/midi",
                    key="download_midi_btn_results"
                )
            if st.session_state.audio_data:
                st.download_button(
                    label="⬇️ Download Audio (WAV)",
                    data=st.session_state.audio_data,
                    file_name=st.session_state.audio_filename,
                    mime="audio/wav",
                    key="download_wav_btn_results"
                )
            
            st.button("🔄 Generate Another", on_click=lambda: st.session_state.update(generated_notes=None, midi_data=None, audio_data=None), key="generate_another_btn")
            
            # Share functionality (mimics script.js share)
            share_text = f"🎵 I just created an AI-generated music composition!\n\nDetails: Notes: {len(st.session_state.generated_notes)}, Creativity: {st.session_state.last_params.get('temperature', 'N/A')}, Tempo: {st.session_state.last_params.get('tempo', 'N/A')} BPM\nNote preview: {', '.join(st.session_state.generated_notes[:10])}...\n\nGenerated with AI Music Generator"
            st.download_button(
                label="🔗 Share Composition",
                data=share_text,
                file_name="ai_music_composition_share.txt",
                mime="text/plain",
                key="share_btn"
            )

# --- AI Model Information Section ---
st.markdown("---") # Separator
with st.container(border=False): # Use container to mimic card
    st.markdown('<div class="card-header-info">📊 AI Model Information</div>', unsafe_allow_html=True)
    
    if metadata_loaded:
        col_acc, col_params, col_vocab, col_epochs = st.columns(4)
        with col_acc:
            st.metric(label="Accuracy", value=f"{metadata_loaded.get('training_info', {}).get('validation_accuracy', 0.0)*100:.1f}%", delta=None, delta_color="off")
        with col_params:
            st.metric(label="Parameters", value=f"{metadata_loaded.get('model_architecture', {}).get('parameters', 0):,}", delta=None, delta_color="off")
        with col_vocab:
            st.metric(label="Vocabulary", value=f"{metadata_loaded.get('vocab_size', 0)} elements", delta=None, delta_color="off")
        with col_epochs:
            st.metric(label="Training Epochs", value=f"{metadata_loaded.get('training_info', {}).get('epochs_trained', 0)}", delta=None, delta_color="off")
        
        st.markdown("---")
        st.markdown("#### 🤖 How It Works")
        st.markdown("""
            This AI music generator uses a deep learning LSTM (Long Short-Term Memory) neural network 
            trained on classical music patterns to compose original melodies.
        """)
        st.markdown("##### Technology Stack:")
        st.markdown("""
            - **Deep Learning:** TensorFlow & Keras
            - **Music Processing:** Music21 library
            - **Web Interface:** Streamlit (replaces Flask & Bootstrap)
            - **Audio Conversion:** Python's `wave` module (replaces FluidSynth for simple browser audio)
        """)

        st.markdown("---")
        st.markdown("#### 💡 Parameter Guide")
        col_guide1, col_guide2, col_guide3 = st.columns(3)
        with col_guide1:
            st.markdown("###### Number of Notes")
            st.markdown("<small>Controls the length of your composition. More notes = longer melody.</small>", unsafe_allow_html=True)
        with col_guide2:
            st.markdown("###### Creativity Level")
            st.markdown("<small>Lower values follow training patterns closely. Higher values are more experimental.</small>", unsafe_allow_html=True)
        with col_guide3:
            st.markdown("###### Tempo")
            st.markdown("<small>Beats per minute. Affects playback speed of your composition.</small>", unsafe_allow_html=True)
        
        if st.checkbox("Show Raw Model Metadata", key="show_raw_metadata_about"):
            st.json(metadata_loaded)
    else:
        st.info("Model metadata could not be loaded.")

# Add a simple footer
st.markdown("---")
st.markdown('<div class="footer-streamlit">&copy; 2025 AI Music Generator - Powered by Deep Learning<br>Created by Uday Reddy</div>', unsafe_allow_html=True)

