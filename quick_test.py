#!/usr/bin/env python3
"""
Quick test script to verify the NLP sentence autocompletion system is working.
Tests core functionality and reports results clearly.
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.autocompletion import SentenceCompleter
from src.utils.data_loader import create_comprehensive_training_data


def test_basic_functionality():
    """Test basic n-gram model functionality."""
    print("🧪 Testing Basic N-gram Functionality")
    print("=" * 50)
    
    try:
        # Initialize completer
        completer = SentenceCompleter(model_type='ngram')
        print("✅ SentenceCompleter initialized")
        
        # Load comprehensive training data
        training_data = create_comprehensive_training_data()
        print(f"✅ Loaded {len(training_data)} training sentences")
        
        # Train model
        completer.train(training_data)
        print("✅ Model training completed")
        
        # Test completion with known good prompts
        test_prompts = ["The weather today", "I love to eat", "Technology"]
        
        for test_text in test_prompts:
            suggestions = completer.complete(test_text, num_suggestions=4)
            
            # Extract next words for diversity check
            next_words = []
            prompt_words = test_text.split()
            
            for suggestion in suggestions:
                words = suggestion.split()
                if len(words) > len(prompt_words):
                    next_word = words[len(prompt_words)]
                    if next_word not in next_words:
                        next_words.append(next_word)
            
            print(f"✅ '{test_text}' → {len(next_words)} diverse options: {next_words[:4]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_model_persistence():
    """Test model saving and loading."""
    print("\n💾 Testing Model Persistence")
    print("=" * 50)
    
    try:
        # Train a model
        completer = SentenceCompleter(model_type='ngram')
        training_data = create_comprehensive_training_data()
        completer.train(training_data)
        
        # Save model
        model_path = 'models/test_model.pkl'
        os.makedirs('models', exist_ok=True)
        completer.save_model(model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Load model
        new_completer = SentenceCompleter(model_type='ngram')
        new_completer.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Test loaded model
        test_text = "Students learn"
        suggestions = new_completer.complete(test_text, num_suggestions=3)
        print(f"✅ Loaded model generated {len(suggestions)} suggestions")
        
        # Show sample result
        if suggestions:
            print(f"   Sample: '{suggestions[0]}'")
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            print("✅ Test model file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Model persistence test failed: {e}")
        return False


def test_existing_models():
    """Test loading and using existing trained models."""
    print("\n📁 Testing Existing Models")
    print("=" * 50)
    
    model_paths = [
        'models/ultimate_model.pkl',
        'models/diverse_trigram_model.pkl',
        'models/grammatical_trigram_model.pkl',
        'models/industrial_trigram_model.pkl'
    ]
    
    found_models = 0
    working_models = 0
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            found_models += 1
            try:
                completer = SentenceCompleter(model_type='ngram')
                completer.load_model(model_path)
                
                # Quick test
                suggestions = completer.complete("The weather today", num_suggestions=3)
                
                model_name = os.path.basename(model_path)
                print(f"✅ {model_name} - Working ({len(suggestions)} suggestions)")
                working_models += 1
                
            except Exception as e:
                model_name = os.path.basename(model_path)
                print(f"❌ {model_name} - Error: {e}")
    
    if found_models == 0:
        print("⚠️  No pre-trained models found. Run 'python train.py' to create models.")
        return None
    else:
        print(f"\n📊 Found {found_models} models, {working_models} working correctly")
        return working_models > 0


def test_diversity_quality():
    """Test the quality and diversity of completions."""
    print("\n🎯 Testing Completion Quality & Diversity")
    print("=" * 50)
    
    try:
        # Use the best available model or train a new one
        completer = None
        
        # Try to load existing model
        model_paths = ['models/ultimate_model.pkl', 'models/diverse_trigram_model.pkl']
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    completer = SentenceCompleter(model_type='ngram')
                    completer.load_model(model_path)
                    print(f"✅ Using existing model: {model_path}")
                    break
                except:
                    continue
        
        # Train new model if no existing model found
        if not completer:
            print("⚠️  No existing models found. Training new model...")
            completer = SentenceCompleter(model_type='ngram')
            training_data = create_comprehensive_training_data()
            completer.train(training_data)
            print("✅ New model trained")
        
        # Test diversity with key prompts
        test_prompts = [
            "The weather today",
            "I love to eat", 
            "Technology",
            "Students learn"
        ]
        
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
            
            diversity_score = len(next_words)
            
            if diversity_score >= 4:
                status = "🎉 EXCELLENT"
                excellent_count += 1
            elif diversity_score >= 2:
                status = "✅ GOOD"
            else:
                status = "⚠️  NEEDS WORK"
            
            print(f"'{prompt:20}' → {diversity_score} options {status}")
        
        success_rate = (excellent_count / len(test_prompts)) * 100
        print(f"\n📊 Quality Score: {success_rate:.1f}% excellent ({excellent_count}/{len(test_prompts)})")
        
        return success_rate >= 50  # At least 50% should be excellent
        
    except Exception as e:
        print(f"❌ Quality test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Quick Test Suite for NLP Sentence Autocompletion")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Model Persistence", test_model_persistence),
        ("Existing Models", test_existing_models),
        ("Quality & Diversity", test_diversity_quality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        result = test_func()
        results[test_name] = result
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
            passed += 1
        elif result is False:
            status = "❌ FAILED"
            failed += 1
        else:
            status = "⚠️  SKIPPED"
            skipped += 1
        
        print(f"{test_name:25} {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All critical tests passed! System is working correctly.")
        print("Your sentence completion system is ready for use!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the error messages above.")
    
    return failed == 0


def main():
    """Main function to run quick tests."""
    success = run_all_tests()
    
    if success:
        print("\n💡 Next steps:")
        print("  - Run 'python example_usage.py' for detailed examples")
        print("  - Run 'python train.py --evaluate' to train new models")
        print("  - Run 'python inference.py --interactive' for interactive use")
        print("  - Check README.md for full documentation")
    else:
        print("\n🔧 Troubleshooting:")
        print("  - Make sure dependencies are installed: pip install -r requirements.txt")
        print("  - Check that the src/ directory structure is intact")
        print("  - Try running 'python train.py' to create new models")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
