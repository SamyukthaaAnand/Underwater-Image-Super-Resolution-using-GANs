from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import math
import time
from enhancer import UnderwaterEnhancer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

class MetricCalculator:
    def __init__(self):
        pass
    
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        return 20 * math.log10(255.0 / math.sqrt(mse)) if mse != 0 else float('inf')
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1*img2, -1, window) - mu1_mu2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    def calculate_uiqm(self, img):
        """Calculate UIQM score for an image"""
        try:
            if len(img.shape) == 2:  # if grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # UICM (Colorfulness)
            rg = np.abs(a - b)
            yb = np.abs(0.5*(a + b) - l)
            uicm = np.mean(rg) * np.mean(yb)
            
            # UISM (Sharpness)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            uism = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # UIConM (Contrast)
            uiconm = np.std(l) / np.mean(l)
            
            return 0.0282 * uicm + 0.2953 * uism + 0.4575 * uiconm
        except Exception as e:
            print(f"UIQM calculation error: {e}")
            return 0.0

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize components
try:
    enhancer = UnderwaterEnhancer()
    metric_calc = MetricCalculator()
except Exception as e:
    print(f"Initialization error: {e}")
    enhancer = None

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default metrics structure
    default_metrics = {
        'input': {'psnr': 0, 'ssim': 0, 'uiqm': 0},
        'output': {'psnr': 0, 'ssim': 0, 'uiqm': 0, 'improvement': 0}
    }
    
    if enhancer is None:
        return render_template('index.html', 
                           error="Service unavailable. Please try again later.",
                           metrics=default_metrics)
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', 
                               error="No file selected",
                               metrics=default_metrics)
            
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', 
                               error="No file selected",
                               metrics=default_metrics)
        
        if not allowed_file(file.filename):
            return render_template('index.html',
                               error="Invalid file type. Allowed: PNG, JPG, JPEG, BMP",
                               metrics=default_metrics)
        
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
            
            # Generate unique filename
            timestamp = int(time.time())
            original_filename = f"{timestamp}_{file.filename}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(input_path)
            
            output_filename = f"enhanced_{original_filename}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            
            # Enhance image
            if not enhancer.enhance_image(input_path, output_path):
                return render_template('index.html',
                                   error="Image enhancement failed",
                                   metrics=default_metrics)
            
            # Calculate metrics
            original_img = cv2.imread(input_path)
            enhanced_img = cv2.imread(output_path)
            
            if original_img is None or enhanced_img is None:
                return render_template('index.html',
                                   error="Could not read images",
                                   metrics=default_metrics)
            
            metrics = {
                'input': {
                    'psnr': metric_calc.calculate_psnr(original_img, original_img),
                    'ssim': metric_calc.calculate_ssim(original_img, original_img),
                    'uiqm': metric_calc.calculate_uiqm(original_img)
                },
                'output': {
                    'psnr': metric_calc.calculate_psnr(original_img, enhanced_img),
                    'ssim': metric_calc.calculate_ssim(original_img, enhanced_img),
                    'uiqm': metric_calc.calculate_uiqm(enhanced_img),
                    'improvement': 0  # Will be calculated below
                }
            }
            
            # Calculate improvement percentage
            if metrics['input']['uiqm'] > 0:
                metrics['output']['improvement'] = round(
                    (metrics['output']['uiqm'] - metrics['input']['uiqm']) / metrics['input']['uiqm'] * 100, 
                    2
                )
            
            return render_template('index.html',
                               original=original_filename,
                               enhanced=output_filename,
                               metrics=metrics)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return render_template('index.html',
                               error=f"Processing error: {str(e)}",
                               metrics=default_metrics)
    
    # GET request - return default page
    return render_template('index.html', metrics=default_metrics)

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Port {port} is in use. Trying port {port+1}...")
            app.run(host='0.0.0.0', port=port+1, debug=True, use_reloader=True)
        else:
            raise