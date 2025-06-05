#!/usr/bin/env python3
"""
Main training script for NLP sentence autocompletion models.

This script trains n-gram models for sentence completion using comprehensive
training data. It supports different model orders (bigram, trigram, 4-gram)
and saves the best performing model.

Usage:
    python train.py [--model-order N] [--output-path PATH]

Example:
    python train.py --model-order 3 --output-path models/best_model.pkl
"""

import argparse
import logging
import os
import sys
from typing import List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.autocompletion import SentenceCompleter
from src.utils.data_loader import create_comprehensive_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(model_order: int = 3, output_path: str = None) -> SentenceCompleter:
    """
    Train a sentence completion model.
    
    Args:
        model_order: N-gram order (2=bigram, 3=trigram, 4=4-gram)
        output_path: Path to save the trained model
        
    Returns:
        Trained SentenceCompleter instance
    """
    logger.info(f"Starting training for {model_order}-gram model")
    
    # Create training data
    logger.info("Loading comprehensive training data...")
    training_data = create_comprehensive_training_data()
    logger.info(f"Loaded {len(training_data)} training sentences")
    
    # Initialize model
    completer = SentenceCompleter(model_type='ngram')
    completer.model.n = model_order
    completer.model.alpha = 0.001  # Low smoothing for better precision
    
    # Train the model
    logger.info("Training model...")
    completer.train(training_data)
    
    # Log model statistics
    vocab_size = len(completer.model.vocabulary)
    context_count = len(completer.model.ngram_counts)
    logger.info(f"Training complete!")
    logger.info(f"  Vocabulary size: {vocab_size} words")
    logger.info(f"  N-gram contexts: {context_count}")
    logger.info(f"  Model order: {model_order}-gram")
    
    # Save model if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        completer.save_model(output_path)
        logger.info(f"Model saved to: {output_path}")
    
    return completer


def evaluate_model(completer: SentenceCompleter) -> None:
    """
    Evaluate the trained model with test prompts.
    
    Args:
        completer: Trained SentenceCompleter instance
    """
    logger.info("Evaluating model performance...")
    
    # Test prompts for evaluation
    test_prompts = [
        "The weather today",
        "I love to eat",
        "Technology has",
        "Students learn",
        "Scientists are",
        "Education is",
        "People enjoy",
        "Innovation drives"
    ]
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    excellent_count = 0
    
    for prompt in test_prompts:
        suggestions = completer.complete(prompt, num_suggestions=6)
        
        # Extract next words
        next_words = []
        prompt_words = prompt.split()
        
        for suggestion in suggestions:
            words = suggestion.split()
            if len(words) > len(prompt_words):
                next_word = words[len(prompt_words)]
                if next_word not in next_words:
                    next_words.append(next_word)
        
        # Evaluate diversity
        diversity_score = len(next_words)
        if diversity_score >= 4:
            status = "✅ EXCELLENT"
            excellent_count += 1
        elif diversity_score >= 2:
            status = "⚠️  GOOD"
        else:
            status = "❌ NEEDS WORK"
        
        print(f"'{prompt:20}' → {next_words[:6]} {status}")
    
    # Summary
    success_rate = (excellent_count / len(test_prompts)) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% ({excellent_count}/{len(test_prompts)} excellent)")
    
    if success_rate >= 75:
        print("🎉 OUTSTANDING MODEL PERFORMANCE!")
    elif success_rate >= 50:
        print("✅ GOOD MODEL PERFORMANCE!")
    else:
        print("⚠️  Model needs more training data or tuning.")


def main():
    """Main function for training script."""
    parser = argparse.ArgumentParser(
        description="Train NLP sentence autocompletion model"
    )
    parser.add_argument(
        '--model-order', 
        type=int, 
        default=3,
        choices=[2, 3, 4, 5],
        help='N-gram model order (default: 3 for trigram)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='models/trained_model.pkl',
        help='Path to save trained model (default: models/trained_model.pkl)'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model after training'
    )
    
    args = parser.parse_args()
    
    print("🚀 NLP Sentence Autocompletion Model Training")
    print("="*60)
    
    try:
        # Train the model
        completer = train_model(
            model_order=args.model_order,
            output_path=args.output_path
        )
        
        # Evaluate if requested
        if args.evaluate:
            evaluate_model(completer)
        
        print("\n🎉 Training completed successfully!")
        print(f"Model saved to: {args.output_path}")
        print("\nTo use the model:")
        print(f"  python inference.py --model-path {args.output_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
