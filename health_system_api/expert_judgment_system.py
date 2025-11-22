#Author:yifan zhu
#check readme.txt

import os
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
import time

# Emotion mapping and conversion
class EmotionMapper:
    # Voice emotion labels
    VOICE_EMOTIONS = {
        0: "female_angry", 1: "female_calm", 2: "female_fearful", 
        3: "female_happy", 4: "female_sad", 5: "male_angry", 
        6: "male_calm", 7: "male_fearful", 8: "male_happy", 9: "male_sad"
    }
    
    # Face emotion labels
    FACE_EMOTIONS = {
        0: 'Angry', 1: 'Disgust', 2: 'Fear', 
        3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
    }
    
    # Standardized emotion categories (for fusion)
    STANDARD_EMOTIONS = [
        'angry', 'calm', 'fearful', 'happy', 'sad', 'neutral', 'surprised', 'disgusted'
    ]
    
    # Voice emotion to standard emotion mapping
    VOICE_TO_STANDARD = {
        "female_angry": "angry", "female_calm": "calm", "female_fearful": "fearful",
        "female_happy": "happy", "female_sad": "sad", "male_angry": "angry",
        "male_calm": "calm", "male_fearful": "fearful", "male_happy": "happy", "male_sad": "sad"
    }
    
    # Face emotion to standard emotion mapping
    FACE_TO_STANDARD = {
        "Angry": "angry", "Disgust": "disgusted", "Fear": "fearful",
        "Happy": "happy", "Neutral": "neutral", "Sad": "sad", "Surprise": "surprised"
    }
    
    @classmethod
    def voice_to_standard(cls, voice_emotion: str) -> str:
        """Convert voice emotion to standard emotion"""
        return cls.VOICE_TO_STANDARD.get(voice_emotion, "neutral")
    
    @classmethod
    def face_to_standard(cls, face_emotion: str) -> str:
        """Convert face emotion to standard emotion"""
        return cls.FACE_TO_STANDARD.get(face_emotion, "neutral")
    
    @classmethod
    def get_emotion_index(cls, emotion: str) -> int:
        """Get the index of standard emotion"""
        try:
            return cls.STANDARD_EMOTIONS.index(emotion.lower())
        except ValueError:
            return cls.STANDARD_EMOTIONS.index("neutral")  # Default to neutral

# Emotion advice generator
class EmotionAdvisor:
    # Emotion advice mapping
    EMOTION_ADVICE = {
        'angry': {
            'comfort': [
                "I notice you seem a bit angry. Deep breathing can help you calm down.",
                "Anger is a normal emotional response, but remember not to let it control your actions.",
                "I understand your anger. Would you like to try shifting your attention to something you enjoy?"
            ],
            'suggestion': [
                "Try counting to 10 to give yourself time to cool down.",
                "Physical activity can help release negative emotions, such as brisk walking or running.",
                "Write down what's making you angry, then think about possible solutions."
            ]
        },
        'sad': {
            'comfort': [
                "I can sense your sadness. This is a completely normal emotional response.",
                "When you're sad, allow yourself to feel the emotion rather than suppressing it.",
                "Remember, there are no permanent cloudy days. The sun will shine again."
            ],
            'suggestion': [
                "Listening to light and cheerful music might improve your mood.",
                "Talking with friends or family can help you move past sadness.",
                "Spending time outdoors in the sunlight can boost your mood."
            ]
        },
        'fearful': {
            'comfort': [
                "Feeling afraid is a natural human response. It means your defense mechanisms are working.",
                "Facing fear requires courage, and you've already taken the first step.",
                "Remember, most of the things we fear never actually happen."
            ],
            'suggestion': [
                "Try focusing on the present moment. Deep breathing exercises can reduce anxiety.",
                "Write down your fears and then analyze how realistic they are.",
                "Progressive relaxation techniques can help relieve physical tension."
            ]
        },
        'happy': {
            'comfort': [
                "It's wonderful to see you so happy! Positive emotions have many health benefits.",
                "Happiness is an emotion worth cherishing. I hope this feeling continues.",
                "Your happiness is contagious!"
            ],
            'suggestion': [
                "Record what makes you happy to create a 'happiness journal'.",
                "Sharing your happiness with others can make this emotion last longer.",
                "Use this positive emotional state to accomplish things you've wanted to do."
            ]
        },
        'neutral': {
            'comfort': [
                "A calm emotional state is a good time for reflection and planning.",
                "Emotional balance is an important component of mental health.",
                "Neutral emotions allow us to view things more objectively."
            ],
            'suggestion': [
                "This is a good time for meditation or mindfulness practice.",
                "You can use this calm period to think about your goals and plans.",
                "Try some creative activities like drawing, writing, or music."
            ]
        },
        'surprised': {
            'comfort': [
                "Surprise indicates you've encountered something unexpected, which is a normal reaction.",
                "Surprise helps us quickly adapt to new situations.",
                "The feeling of surprise usually passes quickly. Give yourself time to adjust."
            ],
            'suggestion': [
                "Take a deep breath and give yourself time to process this unexpected situation.",
                "Consider the positive impacts this surprising event might bring.",
                "Sharing your surprising experience with others can help you understand it better."
            ]
        },
        'disgusted': {
            'comfort': [
                "Disgust is a protection mechanism that helps us avoid potentially harmful substances or situations.",
                "This emotional response is completely normal. No need to feel uncomfortable about it.",
                "Feelings of disgust typically diminish over time."
            ],
            'suggestion': [
                "Try shifting your attention to pleasant things.",
                "If it's a specific thing causing disgust, try gradual exposure to reduce sensitivity.",
                "Deep breathing and relaxation techniques can help alleviate discomfort from disgust."
            ]
        },
        'calm': {
            'comfort': [
                "A calm state is beneficial for both mind and body. This is a great emotional state.",
                "Staying calm helps you think more clearly and make better decisions.",
                "Calmness is a sign of inner strength. Continue maintaining this state."
            ],
            'suggestion': [
                "Use this calm state for activities that require focus.",
                "Meditation or mindfulness practice can help deepen this sense of calm.",
                "Record what makes you feel calm for future reference."
            ]
        }
    }
    
    @classmethod
    def get_random_advice(cls, emotion: str) -> Dict[str, str]:
        """Get random advice based on emotion"""
        emotion = emotion.lower()
        if emotion not in cls.EMOTION_ADVICE:
            emotion = 'neutral'  # Default to neutral
        
        advice = cls.EMOTION_ADVICE[emotion]
        comfort = np.random.choice(advice['comfort'])
        suggestion = np.random.choice(advice['suggestion'])
        
        return {
            'comfort': comfort,
            'suggestion': suggestion
        }

# Expert judgment system
class ExpertJudgmentSystem:
    def __init__(self, voice_weight: float = 0.5, face_weight: float = 0.5):
        """
        Initialize expert judgment system
        
        Parameters:
            voice_weight: Weight for voice emotion
            face_weight: Weight for face emotion
        """
        self.voice_weight = voice_weight
        self.face_weight = face_weight
        self.cache = {}  # Simple caching mechanism
        self.cache_timeout = 60  # Cache timeout (seconds)
    
    def analyze(self, voice_emotion: Optional[str] = None, voice_probs: Optional[List[float]] = None,
                face_emotion: Optional[str] = None, face_probs: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze emotions and generate advice
        
        Parameters:
            voice_emotion: Voice emotion label
            voice_probs: Voice emotion probabilities
            face_emotion: Face emotion label
            face_probs: Face emotion probabilities
            
        Returns:
            Analysis result dictionary
        """
        # Generate cache key
        cache_key = f"{voice_emotion}_{face_emotion}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, cache_result = self.cache[cache_key]
            if time.time() - cache_time < self.cache_timeout:
                return cache_result
        
        # Standardize emotions
        std_voice_emotion = None
        std_face_emotion = None
        
        if voice_emotion:
            std_voice_emotion = EmotionMapper.voice_to_standard(voice_emotion)
        
        if face_emotion:
            std_face_emotion = EmotionMapper.face_to_standard(face_emotion)
        
        # Emotion consistency check
        is_consistent = (std_voice_emotion == std_face_emotion) if (std_voice_emotion and std_face_emotion) else False
        
        # Fuse emotions
        final_emotion = self._fuse_emotions(std_voice_emotion, std_face_emotion)
        
        # Generate advice
        advice = EmotionAdvisor.get_random_advice(final_emotion)
        
        # Build result
        result = {
            'timestamp': time.time(),
            'voice_emotion': voice_emotion,
            'face_emotion': face_emotion,
            'standard_voice_emotion': std_voice_emotion,
            'standard_face_emotion': std_face_emotion,
            'is_consistent': is_consistent,
            'final_emotion': final_emotion,
            'comfort': advice['comfort'],
            'suggestion': advice['suggestion']
        }
        
        # Update cache
        self.cache[cache_key] = (time.time(), result)
        
        return result
    
    def _fuse_emotions(self, voice_emotion: Optional[str], face_emotion: Optional[str]) -> str:
        """
        Fuse voice and face emotions
        
        Parameters:
            voice_emotion: Standardized voice emotion
            face_emotion: Standardized face emotion
            
        Returns:
            Fused emotion
        """
        # If only one emotion is available
        if not voice_emotion and face_emotion:
            return face_emotion
        elif voice_emotion and not face_emotion:
            return voice_emotion
        elif not voice_emotion and not face_emotion:
            return "neutral"  # Default to neutral
        
        # If both emotions are the same
        if voice_emotion == face_emotion:
            return voice_emotion
        
        # Emotion priority (for inconsistent cases)
        # Negative emotions take priority over positive ones, as they are more likely to need attention
        priority_order = ['angry', 'fearful', 'sad', 'disgusted', 'surprised', 'neutral', 'calm', 'happy']
        
        voice_priority = priority_order.index(voice_emotion) if voice_emotion in priority_order else len(priority_order)
        face_priority = priority_order.index(face_emotion) if face_emotion in priority_order else len(priority_order)
        
        # Choose the emotion with higher priority (smaller index)
        if voice_priority <= face_priority:
            return voice_emotion
        else:
            return face_emotion
    
    def update_weights(self, voice_weight: float, face_weight: float) -> None:
        """
        Update emotion weights
        
        Parameters:
            voice_weight: Weight for voice emotion
            face_weight: Weight for face emotion
        """
        self.voice_weight = voice_weight
        self.face_weight = face_weight
        self.cache = {}  # Clear cache

# Test expert judgment system
if __name__ == "__main__":
    expert_system = ExpertJudgmentSystem()
    
    # Test case 1: Voice and face emotions are consistent
    result1 = expert_system.analyze(
        voice_emotion="female_angry",
        face_emotion="Angry"
    )
    print("Case 1 (Consistent emotions):")
    print(json.dumps(result1, indent=2))
    
    # Test case 2: Voice and face emotions are inconsistent
    result2 = expert_system.analyze(
        voice_emotion="male_happy",
        face_emotion="Sad"
    )
    print("\nCase 2 (Inconsistent emotions):")
    print(json.dumps(result2, indent=2))
    
    # Test case 3: Only voice emotion available
    result3 = expert_system.analyze(
        voice_emotion="female_fearful"
    )
    print("\nCase 3 (Voice emotion only):")
    print(json.dumps(result3, indent=2))
    
    # Test case 4: Only face emotion available
    result4 = expert_system.analyze(
        face_emotion="Happy"
    )
    print("\nCase 4 (Face emotion only):")
    print(json.dumps(result4, indent=2))
