# enhancer.py
import cv2
import requests

class UnderwaterEnhancer:
    def __init__(self):
        self.api_key = 'quickstart-QUdJIGlzIGNvbWluZy4uLi4K'
    
    def enhance_image(self, image_path, output_path):
        try:
            # Try API enhancement first
            with open(image_path, 'rb') as f:
                response = requests.post(
                    'https://api.deepai.org/api/torch-srgan',
                    files={'image': f},
                    headers={'api-key': self.api_key}
                )
            enhanced_url = response.json()['output_url']
            
            # Download and save enhanced image
            enhanced_data = requests.get(enhanced_url).content
            with open(output_path, 'wb') as f:
                f.write(enhanced_data)
            return True
            
        except Exception as e:
            print(f"Enhancement failed: {e}")
            # Fallback to basic OpenCV enhancement
            img = cv2.imread(image_path)
            if img is not None:
                # Simple contrast enhancement
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
                cv2.imwrite(output_path, enhanced)
                return True
            return False
    
