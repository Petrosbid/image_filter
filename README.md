# 🖼️ Advanced Image Filtering Studio

> **Professional Image Processing Suite with Mathematical Formulas**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

[![License](https://img.shields.io/github/license/mhmd/image_filter?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/mhmd/image_filter?style=for-the-badge)](https://github.com/mhmd/image_filter/stars)
[![Issues](https://img.shields.io/github/issues/mhmd/image_filter?style=for-the-badge)](https://github.com/mhmd/image_filter/issues)

</div>

---

## 🌟 Overview

**Advanced Image Filtering Studio** is a cutting-edge image processing application that combines state-of-the-art computer vision algorithms with an intuitive user interface. Built on the foundation of Gonzalez's digital image processing concepts, this suite offers comprehensive filtering capabilities with real-time preview and mathematical formula visualization.

## ✨ Key Features

### 🧮 Mathematical Precision
- **LaTeX Formula Rendering** 📐 - View mathematical equations for each filter
- **Academic Foundation** 🎓 - Based on Gonzalez's digital image processing principles
- **Scientific Accuracy** 🔬 - Precise implementation of mathematical models

### 🎨 Professional Filters
- **Intensity Transformations** 🌗 - Negative, Log, Gamma, Contrast Stretching
- **Histogram Processing** 📊 - Equalization, CLAHE, Adaptive techniques
- **Smoothing Filters** 🧼 - Gaussian, Bilateral, Median, Box filters
- **Edge Detection** ⚡ - Sobel, Canny, Laplacian, Prewitt operators
- **Frequency Domain** 🌊 - Lowpass, Highpass, Butterworth, Gaussian filters

### 🎯 Noise Generation
- **Statistical Models** 📈 - Gaussian, Uniform, Exponential distributions
- **Impulse Noise** 💥 - Salt & Pepper, Speckle noise simulation
- **Advanced Distributions** 🧠 - Rayleigh, Poisson, Erlang/Gamma models

### ⚡ Real-time Processing
- **Live Preview** 📹 - Real-time camera feed with filter application
- **WebSocket Integration** 🌐 - Low-latency communication
- **Performance Optimized** 🚀 - Efficient algorithms for smooth experience

## 🛠️ Tech Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| [Python](https://www.python.org/) | Backend Logic | 3.8+ |
| [Flask](https://flask.palletsprojects.com/) | Web Framework | 2.0+ |
| [OpenCV](https://opencv.org/) | Computer Vision | 4.5+ |
| [NumPy](https://numpy.org/) | Numerical Computing | Latest |
| [Flask-SocketIO](https://flask-socketio.readthedocs.io/) | Real-time Communication | 5.0+ |
| [HTML5](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5) | Frontend Structure | Standard |
| [TailwindCSS](https://tailwindcss.com/) | Styling Framework | Latest |
| [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript) | Client-side Logic | ES6+ |
| [MathJax](https://www.mathjax.org/) | LaTeX Rendering | 3.0+ |

## 📦 Installation

### Prerequisites
- 🐍 Python 3.8 or higher
- 📦 pip package manager
- 🖥️ Modern web browser

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/mhmd/image_filter.git
cd image_filter

# Navigate to backend directory
cd Backend

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python app.py
```

### Dependencies Installation

```bash
pip install opencv-python numpy flask flask-cors flask-socketio pillow
```

## 🚀 Usage

### 1. Launch Backend Server
```bash
cd Backend
python app.py
```

### 2. Access Web Interface
- **Main Interface**: [http://localhost:5000/Frontend/html/professional_index.html](http://localhost:5000/Frontend/html/professional_index.html)
- **Real-time Interface**: [http://localhost:5000/Frontend/html/professional_realtime.html](http://localhost:5000/Frontend/html/professional_realtime.html)

### 3. Features Guide
- 📤 **Upload Images** - Drag & drop or browse files
- 🎛️ **Adjust Parameters** - Fine-tune filter settings
- 🧮 **View Formulas** - See mathematical representations
- 📸 **Download Results** - Save processed images
- 📹 **Real-time Preview** - Live camera filtering

## 🎨 Filter Categories

### 🌗 Intensity Transformations
- **Negative** - `s = L - 1 - r`
- **Log Transform** - `s = c · log(1 + r)`
- **Gamma Correction** - `s = c · r^γ`

### 📊 Histogram Processing
- **Equalization** - `sk = Σ(j=0 to k) nj/N · (L-1)`
- **CLAHE** - Adaptive histogram equalization

### 🧼 Smoothing Filters
- **Gaussian Blur** - `H(u,v) = e^(-(u²+v²)/(2σ²))`
- **Bilateral Filter** - Edge-preserving smoothing
- **Median Filter** - Noise reduction

### ⚡ Edge Detection
- **Sobel Operator** - Gradient-based detection
- **Canny Edge** - Optimal edge detection
- **Laplacian** - Second derivative operator

### 🌊 Frequency Domain
- **Lowpass Filters** - Ideal, Butterworth, Gaussian
- **Highpass Filters** - For edge enhancement

### 📈 Noise Models
- **Gaussian** - Normal distribution noise
- **Salt & Pepper** - Impulse noise
- **Poisson** - Photon-based noise
- **Rayleigh** - Asymmetric distribution

## 🏗️ Project Structure

```
image_filter/
├── Backend/
│   ├── app.py                 # Main Flask application
│   ├── main_filters.py       # Comprehensive image processor
│   ├── main_Noises.py        # Noise generation algorithms
│   ├── Additional_filters.py # Extra filter implementations
│   └── realtime_filters.py   # Real-time processing
├── Frontend/
│   ├── html/
│   │   ├── professional_index.html    # Main interface
│   │   └── professional_realtime.html # Real-time interface
│   ├── css/
│   │   └── styles.css               # Tailwind customization
│   └── javascript/
│       └── script.js                # Client-side logic
├── requirements.txt          # Python dependencies
└── README.md               # Documentation
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. ✅ **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. ⬆️ **Push** to the branch (`git push origin feature/amazing-feature`)
5. 🔄 **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 coding standards
- Write comprehensive docstrings
- Maintain mathematical accuracy
- Test all filter implementations
- Update documentation as needed


<div align="center">

**Built with ❤️**

⭐ Star this repo if you find it helpful!

</div>