"""
Main interface for the sentence autocompletion system.
Provides a unified API for both n-gram and transformer-based models.
"""

from typing import List, Optional, Union
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseCompleter(ABC):
    """Abstract base class for sentence completion models."""
    
    @abstractmethod
    def complete(self, partial_sentence: str, num_suggestions: int = 3) -> List[str]:
        """
        Complete a partial sentence.
        
        Args:
            partial_sentence: The incomplete sentence to complete
            num_suggestions: Number of completion suggestions to return
            
        Returns:
            List of completion suggestions
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a trained model from file."""
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """Save the current model to file."""
        pass


class SentenceCompleter:
    """
    Main sentence completion interface that can use different underlying models.
    """
    
    def __init__(self, model_type: str = 'ngram', model_path: Optional[str] = None):
        """
        Initialize the sentence completer.
        
        Args:
            model_type: Type of model to use ('ngram' or 'transformer')
            model_path: Path to a pre-trained model (optional)
        """
        self.model_type = model_type
        self.model = None
        
        if model_type == 'ngram':
            from .ngram.model import NGramCompleter
            self.model = NGramCompleter()
        elif model_type == 'transformer':
            from .transformer.model import TransformerCompleter
            self.model = TransformerCompleter()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if model_path:
            self.load_model(model_path)
            
        logger.info(f"Initialized {model_type} sentence completer")
    
    def complete(self, partial_sentence: str, num_suggestions: int = 3) -> List[str]:
        """
        Get completion suggestions for a partial sentence.
        
        Args:
            partial_sentence: The incomplete sentence
            num_suggestions: Number of suggestions to return
            
        Returns:
            List of completion suggestions
        """
        if not self.model:
            raise RuntimeError("No model loaded")
        
        # Clean and validate input
        partial_sentence = partial_sentence.strip()
        if not partial_sentence:
            return []
        
        logger.info(f"Completing: '{partial_sentence}'")
        suggestions = self.model.complete(partial_sentence, num_suggestions)
        
        return suggestions
    
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model."""
        if not self.model:
            raise RuntimeError("No model initialized")
        
        self.model.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    
    def save_model(self, model_path: str) -> None:
        """Save the current model."""
        if not self.model:
            raise RuntimeError("No model initialized")
        
        self.model.save_model(model_path)
        logger.info(f"Saved model to {model_path}")
    
    def train(self, training_data: Union[str, List[str]], **kwargs) -> None:
        """
        Train the model on provided data.
        
        Args:
            training_data: Either a file path or list of sentences
            **kwargs: Additional training parameters
        """
        if not self.model:
            raise RuntimeError("No model initialized")
        
        if hasattr(self.model, 'train'):
            self.model.train(training_data, **kwargs)
            logger.info("Model training completed")
        else:
            raise NotImplementedError(f"Training not implemented for {self.model_type}")
    
    def evaluate(self, test_data: Union[str, List[str]], **kwargs) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Either a file path or list of sentences
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.model:
            raise RuntimeError("No model initialized")
        
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(test_data, **kwargs)
        else:
            raise NotImplementedError(f"Evaluation not implemented for {self.model_type}")


def demo():
    """Simple demonstration of the sentence completion system."""
    print("Sentence Autocompletion System Demo")
    print("=" * 40)
    
    # Example with n-gram model (placeholder)
    try:
        completer = SentenceCompleter(model_type='ngram')
        
        test_sentences = [
            "The weather today is",
            "I love to eat",
            "Machine learning is",
            "The quick brown fox"
        ]
        
        for sentence in test_sentences:
            print(f"\nInput: '{sentence}'")
            try:
                suggestions = completer.complete(sentence, num_suggestions=3)
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
            except Exception as e:
                print(f"  Error: {e}")
                
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Note: This is expected if models haven't been implemented yet.")


if __name__ == "__main__":
    demo()
