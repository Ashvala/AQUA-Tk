from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import warnings

@dataclass
class PEAQResult:
    """Results from PEAQ analysis"""
    odg: float  # Objective Difference Grade
    di: float   # Distortion Index  
    mov: Dict[str, float]  # Model Output Variables
    
    def __repr__(self):
        return f"""PEAQ Analysis Results:
        ODG (Objective Difference Grade): {self.odg:.3f} 
        DI (Distortion Index): {self.di:.3f}
        
        Model Output Variables:
        ----------------------
        {self._format_mov()}
        """
    
    def _format_mov(self) -> str:
        return "\n        ".join(f"{k}: {v:.3f}" for k,v in self.mov.items())

class PEAQ:
    """
    Perceptual Evaluation of Audio Quality (PEAQ) - Basic Version
    Implementation of ITU-R BS.1387-1
    """
    
    def __init__(self, version: str = "basic"):
        """
        Initialize PEAQ analyzer
        
        Args:
            version: Either "basic" or "advanced". Defaults to "basic".
        """
        if version not in ["basic", "advanced"]:
            raise ValueError("Version must be either 'basic' or 'advanced'")
        self.version = version
        self._init_state()
        
    def _init_state(self, num_channels: int = 1):
        """Initialize internal state variables for each channel"""
        # Import init_state from peaq_basic to ensure consistency
        from aquatk.metrics.PEAQ.peaq_basic import init_state
        self.num_channels = num_channels
        self.states = [init_state() for _ in range(num_channels)]
        # Keep backward compatibility
        self.state = self.states[0]

    def analyze_files(self,
                     reference_file: Union[str, Path],
                     test_file: Union[str, Path],
                     progress_bar: bool = True) -> PEAQResult:
        """
        Analyze audio quality between reference and test files

        Supports both mono and stereo audio. For stereo, each channel is
        processed separately and MOVs are averaged (matching C implementation).

        Args:
            reference_file: Path to reference audio file
            test_file: Path to test audio file
            progress_bar: Whether to show progress bar during analysis

        Returns:
            PEAQResult object containing analysis results

        Raises:
            FileNotFoundError: If either audio file cannot be found
            ValueError: If audio files have different sample rates or channels
        """
        # Load audio files
        ref_data, ref_blocks_per_ch, ref_rate, ref_num_ch = self._load_audio(reference_file)
        test_data, test_blocks_per_ch, test_rate, test_num_ch = self._load_audio(test_file)

        # Validate channel count
        if ref_num_ch != test_num_ch:
            raise ValueError(f"Channel count mismatch: reference has {ref_num_ch}, test has {test_num_ch}")

        num_channels = ref_num_ch

        # Reset state for fresh analysis (one state per channel)
        self._init_state(num_channels)

        # Process each channel
        channel_results = []

        for ch in range(num_channels):
            ref_blocks = ref_blocks_per_ch[ch]
            test_blocks = test_blocks_per_ch[ch]
            state = self.states[ch]

            final_movs = None
            final_di = None
            final_odg = None

            desc = f"Channel {ch+1}/{num_channels}" if num_channels > 1 else None
            iterator = tqdm(range(len(ref_blocks)), desc=desc) if progress_bar else range(len(ref_blocks))

            for i in iterator:
                boundaryflag = self._check_boundary(ref_blocks[i], test_blocks[i], ref_rate)
                _, _, movs, di, odg = self._process_block_with_state(
                    ref_blocks[i],
                    test_blocks[i],
                    ref_rate,
                    boundaryflag,
                    test_rate,
                    state
                )
                final_movs = movs
                final_di = di
                final_odg = odg
                state["count"] += 1

            channel_results.append({
                'movs': final_movs,
                'di': final_di,
                'odg': final_odg
            })

        # Average MOVs across channels (like C implementation)
        if num_channels == 1:
            final_movs = channel_results[0]['movs']
            final_di = channel_results[0]['di']
            final_odg = channel_results[0]['odg']
        else:
            # Average the MOVs from both channels
            final_movs = self._average_movs([r['movs'] for r in channel_results])
            # Recompute DI and ODG from averaged MOVs
            from aquatk.metrics.PEAQ.neural import neural
            neural_out = neural(final_movs.to_dict())
            final_di = neural_out['DI']
            final_odg = neural_out['ODG']

        return PEAQResult(
            odg=final_odg,
            di=final_di,
            mov=final_movs.to_dict() if final_movs else {}
        )

    def _process_block_with_state(self, ref_block: np.ndarray, test_block: np.ndarray,
                                   rate: int, boundaryflag: bool, test_rate: int, state: dict):
        """Process a single block of audio with a specific state dict"""
        from aquatk.metrics.PEAQ.peaq_basic import process_audio_block
        return process_audio_block(
            ref_block, test_block,
            rate=rate,
            state=state,
            boundflag=boundaryflag,
            test_rate=test_rate
        )

    def _average_movs(self, movs_list):
        """Average MOVs from multiple channels"""
        from aquatk.metrics.PEAQ.utils import MOV
        avg_movs = MOV()
        mov_names = ['WinModDiff1b', 'AvgModDiff1b', 'AvgModDiff2b', 'RmsNoiseLoudb',
                     'BandwidthRefb', 'BandwidthTestb', 'TotalNMRb', 'RelDistFramesb',
                     'ADBb', 'MFPDb', 'EHSb']
        for name in mov_names:
            values = [getattr(m, name) for m in movs_list]
            setattr(avg_movs, name, sum(values) / len(values))
        return avg_movs
    
    def analyze_arrays(self,
                      reference: np.ndarray,
                      test: np.ndarray,
                      sample_rate: int,
                      progress_bar: bool = True) -> PEAQResult:
        """
        Analyze audio quality between reference and test numpy arrays

        Automatically resamples to 48 kHz if needed (ITU-R BS.1387 requirement).

        Args:
            reference: Reference audio as numpy array
            test: Test audio as numpy array
            sample_rate: Sample rate of audio
            progress_bar: Whether to show progress bar during analysis

        Returns:
            PEAQResult object containing analysis results

        Raises:
            ValueError: If arrays have different shapes or invalid data
        """
        # Reset state for fresh analysis
        self._init_state()

        test = test[:len(reference)]  # Ensure same length
        if reference.shape != test.shape:
            raise ValueError("Reference and test arrays must have same shape")

        # Resample to 48 kHz if needed (ITU-R BS.1387 requirement)
        if sample_rate != PEAQ_SAMPLE_RATE:
            warnings.warn(
                f"Input sample rate {sample_rate} Hz differs from PEAQ standard {PEAQ_SAMPLE_RATE} Hz. "
                f"Resampling to {PEAQ_SAMPLE_RATE} Hz.",
                UserWarning
            )
            # Calculate new number of samples
            num_samples = int(len(reference) * PEAQ_SAMPLE_RATE / sample_rate)
            reference = signal.resample(reference, num_samples)
            test = signal.resample(test, num_samples)
            sample_rate = PEAQ_SAMPLE_RATE

        # Convert to 16-bit integer scale to match C implementation
        # Check if already in integer scale (values > 1.0)
        if np.abs(reference).max() <= 1.0:
            reference = (reference * 32768).astype(np.float64)
            test = (test * 32768).astype(np.float64)

        ref_blocks = self._array_to_blocks(reference)
        test_blocks = self._array_to_blocks(test)

        return self._analyze_blocks(ref_blocks, test_blocks, sample_rate, progress_bar)

    def _load_audio(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, List[List[np.ndarray]], int, int]:
        """Load and prepare audio file for analysis.

        Uses native sample rate (matching C implementation behavior).
        Preserves stereo channels for proper PEAQ processing.

        Returns:
            Tuple of (data, blocks_per_channel, sample_rate, num_channels)
        """
        data, rate = sf.read(str(filepath))

        # Determine number of channels
        if len(data.shape) == 1:
            num_channels = 1
            data = data.reshape(-1, 1)  # Make it (samples, 1) for consistency
        else:
            num_channels = data.shape[1]
            if num_channels > 2:
                raise ValueError(f"PEAQ supports up to 2 channels, got {num_channels}")

        # No resampling - use native sample rate (matches C implementation)
        # Note: ITU-R BS.1387 specifies 48 kHz, but peaqb-fast uses native rate

        # Convert to 16-bit integer scale to match C implementation
        # The C code reads raw PCM samples as signed integers
        data = (data * 32768).astype(np.float64)

        # Create blocks for each channel
        blocks_per_channel = []
        for ch in range(num_channels):
            blocks_per_channel.append(self._array_to_blocks(data[:, ch]))

        return data, blocks_per_channel, rate, num_channels

    def _validate_audio(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_rate: int, test_rate: int):
        """Validate audio files are compatible for comparison"""
        if ref_rate != test_rate:
            raise ValueError(f"Sample rates must match: {ref_rate} != {test_rate}")
        if ref_data.shape != test_data.shape:
            raise ValueError(f"Audio lengths must match: {ref_data.shape} != {test_data.shape}")

    def _array_to_blocks(self, data: np.ndarray, 
                        block_size: int = 2048, 
                        overlap: int = 1024) -> List[np.ndarray]:
        """Convert audio array to overlapping blocks"""
        blocks = []
        start = 0
        while start + block_size <= len(data):
            blocks.append(data[start:start + block_size])
            start += block_size - overlap
        return blocks

    def _check_boundary(self, ref_block: np.ndarray, test_block: np.ndarray, 
                       rate: int, limit: int = 200) -> bool:
        """Check if block contains signal boundary"""
        window = 5
        for k in range(len(ref_block) - window + 1):
            if (np.abs(test_block[k:k+window]).sum() > limit or
                np.abs(ref_block[k:k+window]).sum() > limit):
                return True
        return False
    
    def _process_block(self, ref_block: np.ndarray, test_block: np.ndarray,
                      rate: int, boundaryflag: bool, test_rate: int):
        """Process a single block of audio"""
        # Import internally to avoid circular imports
        from aquatk.metrics.PEAQ.peaq_basic import process_audio_block
        return process_audio_block(
            ref_block, test_block,
            rate=rate,
            state=self.state,
            boundflag=boundaryflag,
            test_rate=test_rate
        )

    def _analyze_blocks(self, ref_blocks, test_blocks, sample_rate, progress_bar=True):
        """Analyze a list of audio blocks"""
        # Note: State should already be initialized by caller (analyze_files or analyze_arrays)
        final_movs = None
        final_di = None
        final_odg = None

        iterator = tqdm(range(len(ref_blocks))) if progress_bar else range(len(ref_blocks))

        for i in iterator:
            boundaryflag = self._check_boundary(ref_blocks[i], test_blocks[i], sample_rate)
            _, _, movs, di, odg = self._process_block(
                ref_blocks[i],
                test_blocks[i],
                sample_rate,
                boundaryflag,
                sample_rate  # Using same rate for test since we validated they match
            )
            # Keep the last frame's values (like the C implementation)
            final_movs = movs
            final_di = di
            final_odg = odg
            self.state["count"] += 1

        # Return the final frame's accumulated results
        # The C implementation outputs the final frame's values which contain
        # the accumulated/averaged MOVs computed over all frames
        return PEAQResult(
            odg=final_odg,
            di=final_di,
            mov=final_movs.to_dict() if final_movs else {}
        )

@dataclass
class PEAQ2FResult:
    """Results from PEAQ 2F model analysis"""
    score: float
    movs: Dict[str, float]

    def __repr__(self):
        return f"""PEAQ 2F Analysis Results:
        Quality Score: {self.score:.3f}
        
        Model Output Variables:
        ----------------------
        AvgModDiff1: {self.movs['AvgModDiff1']:.3f}
        ADB: {self.movs['ADB']:.3f}
        """

class PEAQ2F(PEAQ):
    """
    Simplified PEAQ model using only 2 factors (AvgModDiff1 and ADB)
    """
    
    def __init__(self):
        super().__init__(version="basic")  # We'll still use basic version infrastructure
        
    def _calculate_2f_score(self, avg_mod_diff1: float, adb: float) -> float:
        """
        Calculate the 2F model score using the provided formula
        
        Args:
            avg_mod_diff1: Average modulation difference 1 MOV
            adb: Average Distorted Block MOV
            
        Returns:
            Final quality score
        """
        numerator = 56.1345
        denominator = 1 + (-0.0282 * avg_mod_diff1 - 0.8628)**2
        
        score = (numerator / denominator) - 27.1451 * adb + 86.3515
        bounded_score = np.clip(score, 0, 100)

        return bounded_score
    
    def analyze_files(self, reference_file: str, test_file: str, 
                     progress_bar: bool = True) -> PEAQ2FResult:
        """
        Analyze audio quality using only AvgModDiff1 and ADB MOVs
        """
        # Use parent class to get all MOVs
        peaq_result = super().analyze_files(reference_file, test_file, progress_bar)
        
        # Extract just the MOVs we need
        relevant_movs = {
            'AvgModDiff1': peaq_result.mov['AvgModDiff1b'],
            'ADB': peaq_result.mov['ADBb']
        }
        
        # Calculate 2F score
        score = self._calculate_2f_score(
            avg_mod_diff1=relevant_movs['AvgModDiff1'],
            adb=relevant_movs['ADB']
        )
        
        return PEAQ2FResult(score=score, movs=relevant_movs)
    
    def analyze_arrays(self, reference: np.ndarray, test: np.ndarray,
                      sample_rate: int, progress_bar: bool = True) -> PEAQ2FResult:
        """
        Analyze audio arrays using only AvgModDiff1 and ADB MOVs
        """
        # Use parent class to get all MOVs
        peaq_result = super().analyze_arrays(reference, test, sample_rate, progress_bar)
        
        # Extract just the MOVs we need
        relevant_movs = {
            'AvgModDiff1': peaq_result.mov['AvgModDiff1b'],
            'ADB': peaq_result.mov['ADBb']
        }
        
        # Calculate 2F score
        score = self._calculate_2f_score(
            avg_mod_diff1=relevant_movs['AvgModDiff1'],
            adb=relevant_movs['ADB']
        )
        
        return PEAQ2FResult(score=score, movs=relevant_movs)




if __name__ == "__main__":
    # Initialize PEAQ analyzer
    peaq = PEAQ(version="basic")
    
    # Analyze audio files
    # result = peaq.analyze_files("ref.wav", "test.wav")

    # print(result)
    
    # Or analyze numpy arrays
    ref_audio = np.random.randn(48000)  # 1 second at 48kHz
    test_audio = ref_audio + 0.01 * np.random.randn(48000)  # Add noise
    result = peaq.analyze_arrays(ref_audio, test_audio, sample_rate=48000)
    print(result)
    peaq_2f = PEAQ2F()
    # result = peaq_2f.analyze_files("reference.wav", "test.wav")
    # print(result)
    
    # Or analyze numpy arrays
    ref_audio = np.random.randn(48000)  # 1 second at 48kHz
    test_audio = ref_audio + 0.01 * np.random.randn(48000)  # Add noise
    result = peaq_2f.analyze_arrays(ref_audio, test_audio, sample_rate=48000)
    print(result)