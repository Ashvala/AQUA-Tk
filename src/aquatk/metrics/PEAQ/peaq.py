from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

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
        
    def _init_state(self):
        """Initialize internal state variables"""
        self.state = {
            "countboundary": 1,
            "RelDistFramesb": 0,
            "nmrtmp": 0, 
            "countenergy": 1,
            "EHStmp": 0,
            "nltmp": 0,
            "noise": 0,
            "internal_count": 0,
            "loudcounter": 0,
            "sumBandwidthRefb": 0,
            "sumBandwidthTestb": 0,
            "countref": 0,
            "counttest": 0,
            "BandwidthRefb": 0,
            "BandwidthTestb": 0,
            "count": 1,
            "RelDistTmp": 0,
            "CFFTtemp": 0,
            "Ptildetemp": 0,
            "PMtmp": 0,
            "QSum": 0,
            "ndistorcedtmp": 0,
            "Cffttmp": np.zeros(1024),
        }

    def analyze_files(self, 
                     reference_file: Union[str, Path], 
                     test_file: Union[str, Path],
                     progress_bar: bool = True) -> PEAQResult:
        """
        Analyze audio quality between reference and test files
        
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
        ref_data, ref_blocks, ref_rate = self._load_audio(reference_file)
        test_data, test_blocks, test_rate = self._load_audio(test_file)
        
        # Validate audio files
        # self._validate_audio(ref_data, test_data, ref_rate, test_rate)
        
        # Process blocks
        mov_list = []
        di_list = []
        odg_list = []
        
        iterator = tqdm(range(len(ref_blocks))) if progress_bar else range(len(ref_blocks))
        
        for i in iterator:
            boundaryflag = self._check_boundary(ref_blocks[i], test_blocks[i], ref_rate)
            _, _, movs, di, odg = self._process_block(
                ref_blocks[i], 
                test_blocks[i],
                ref_rate,
                boundaryflag,
                test_rate
            )
            mov_list.append(movs)
            di_list.append(di) 
            odg_list.append(odg)
            self.state["count"] += 1
            
        # Calculate final results
        final_di = np.mean(di_list)
        final_odg = np.mean(odg_list)
        final_mov = self._aggregate_movs(mov_list)
        
        return PEAQResult(
            odg=final_odg,
            di=final_di,
            mov=final_mov
        )
    
    def analyze_arrays(self,
                      reference: np.ndarray,
                      test: np.ndarray, 
                      sample_rate: int,
                      progress_bar: bool = True) -> PEAQResult:
        """
        Analyze audio quality between reference and test numpy arrays
        
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
        test = test[:len(reference)]  # Ensure same length
        if reference.shape != test.shape:
            raise ValueError("Reference and test arrays must have same shape")
          
        ref_blocks = self._array_to_blocks(reference)
        test_blocks = self._array_to_blocks(test)
        
        return self._analyze_blocks(ref_blocks, test_blocks, sample_rate, progress_bar)

    def _load_audio(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, List[np.ndarray], int]:
        """Load and prepare audio file for analysis"""
        data, rate = sf.read(str(filepath))
        if len(data.shape) > 1:  # Stereo to mono
            data = data.mean(axis=1)
        blocks = self._array_to_blocks(data)
        return data, blocks, rate

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

    def _aggregate_movs(self, mov_list: List[Dict]) -> Dict[str, float]:
        """Aggregate MOVs from all blocks"""
        if not mov_list:
            return {}
            
        # Get all MOV keys
        mov_keys = mov_list[0].to_dict().keys()
        
        # Average each MOV across all blocks
        final_movs = {}
        for key in mov_keys:
            values = [mov.to_dict()[key] for mov in mov_list]
            final_movs[key] = np.mean(values)
            
        return final_movs
    
    def _analyze_blocks(self, ref_blocks, test_blocks, sample_rate, progress_bar=True):
        """Analyze a list of audio blocks"""
        mov_list = []
        di_list = []
        odg_list = []
        
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
            mov_list.append(movs)
            di_list.append(di) 
            odg_list.append(odg)
            self.state["count"] += 1
            
        # Calculate final results
        final_di = np.mean(di_list)
        final_odg = np.mean(odg_list)
        final_mov = self._aggregate_movs(mov_list)
        
        return PEAQResult(
            odg=final_odg,
            di=final_di,
            mov=final_mov
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