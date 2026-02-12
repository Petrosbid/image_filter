// Global Variables
let uploadedFilename = null;
let currentFilter = 'original';
let filters = [];
let currentParams = {};
let selectedCategory = 'all';
let originalImageData = null; // The original uploaded image
let currentImageData = null; // The current working image
let isUsingModifiedOriginal = false; // Track if user has set a modified image as original

// API Configuration
const API_URL = 'http://localhost:5000/api';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadFilters();
    setupEventListeners();
    initializeSlider();
});

// Setup event listeners
function setupEventListeners() {
    // File input
    document.getElementById('fileInput').addEventListener('change', handleFileUpload);

    // Click to upload
    const uploadBox = document.getElementById('uploadBox');
    uploadBox.addEventListener('click', function() {
        document.getElementById('fileInput').click();
    });

    // Drag and drop
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);

    // Main category tabs
    document.querySelectorAll('.category-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            document.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');

            const category = this.dataset.category;

            // Hide all submenus
            document.getElementById('subcategoryTabs').classList.add('hidden');
            document.getElementById('otherCategoryTabs').classList.add('hidden');

            // Show appropriate submenu if needed
            if (category === 'main-filters') {
                document.getElementById('subcategoryTabs').classList.remove('hidden');
                // Don't render filters yet, wait for subcategory selection
            } else if (category === 'other') {
                document.getElementById('otherCategoryTabs').classList.remove('hidden');
                // Don't render filters yet, wait for subcategory selection
            } else {
                // For 'all' category, render all filters
                selectedCategory = category;
                renderFilters(selectedCategory);
            }
        });
    });

    // Subcategory tabs
    document.querySelectorAll('.subcategory-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            document.querySelectorAll('.subcategory-tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            selectedCategory = this.dataset.category;
            renderFilters(selectedCategory);
        });
    });

    // Other category tabs
    document.querySelectorAll('.other-category-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            document.querySelectorAll('.other-category-tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            selectedCategory = this.dataset.category;
            renderFilters(selectedCategory);
        });
    });
}

// Load filters from API
async function loadFilters() {
    showLoading(true);
    try {
        const response = await fetch(`${API_URL}/get-filters`);
        const data = await response.json();
        filters = data.filters;
        renderFilters();
        showToast('فیلترها با موفقیت بارگذاری شدند', 'success');
        const uniqueNames = [...new Set(filters.map(item => item.category))];
        console.log(uniqueNames);
    } catch (error) {
        console.error('Error loading filters:', error);
        showToast('خطا در بارگذاری فیلترها', 'error');
    } finally {
        showLoading(false);
    }
}

// Render filters based on category
function renderFilters(category = 'all') {
    const grid = document.getElementById('filtersGrid');
    grid.innerHTML = '';

    let filteredFilters = [];
    if (category === 'all') {
        filteredFilters = filters;
    } else if (category === 'main-filters') {
        // When 'main-filters' is selected (but no specific subcategory), show all main categories
        filteredFilters = filters.filter(f =>
            f.category === 'اصلاح نور و کنتراست' ||
            f.category === 'پردازش هیستوگرام' ||
            f.category === 'کاهش نویز و هموارسازی' ||
            f.category === 'لبه‌یابی و تیزسازی' ||
            f.category === 'پردازش در حوزه فرکانس'
        );
    } else if (category === 'Noise Filters') {
        // When 'Noise Filters' is selected, show all noise filters
        filteredFilters = filters.filter(f => f.category === 'Noise Filters');
    } else {
        // Map Persian category names to English categories from the API
        const categoryMapping = {
            'اصلاح نور و کنتراست': 'اصلاح نور و کنتراست',
            'پردازش هیستوگرام': 'پردازش هیستوگرام',
            'کاهش نویز و هموارسازی': 'کاهش نویز و هموارسازی',
            'لبه‌یابی و تیزسازی': 'لبه‌یابی و تیزسازی',
            'پردازش در حوزه فرکانس': 'پردازش در حوزه فرکانس',
            'Additional': 'Additional',
            'Vintage': 'Vintage', // Changed from 'Additional - Vintage'
            'Basic': 'Additional', // Basic filters are under Additional
            'Color': 'Color', // Direct match
            'BlackWhite': 'BW', // Changed from 'Additional - BW' to 'BW'
            'Blur': 'Blur', // Direct match
            'Edge Detection': 'Edge Detection', // Direct match
            'Artistic': 'Artistic', // Direct match
            'Special': 'Special', // Direct match
            'Technical': 'Technical', // Direct match
            'Distortion': 'Distortion', // Direct match
            'Lighting': 'Lighting', // Direct match
            'Noise Filters': 'Noise Filters' // Direct match
        };

        const mappedCategory = categoryMapping[category] || category;

        // Handle special case for different category types
        if (mappedCategory === 'Additional') {
            filteredFilters = filters.filter(f =>
                f.category === 'Additional' ||
                f.category.startsWith('Additional - ')
            );
        } else if (category === 'main-filters') {
            // When 'main-filters' is selected, show filters from the main scientific categories
            filteredFilters = filters.filter(f =>
                f.category === 'اصلاح نور و کنتراست' ||
                f.category === 'پردازش هیستوگرام' ||
                f.category === 'کاهش نویز و هموارسازی' ||
                f.category === 'لبه‌یابی و تیزسازی' ||
                f.category === 'پردازش در حوزه فرکانس'
            );
        } else if (category === 'other') {
            // When 'other' is selected, show filters from other categories
            filteredFilters = filters.filter(f =>
                f.category !== 'اصلاح نور و کنتراست' &&
                f.category !== 'پردازش هیستوگرام' &&
                f.category !== 'کاهش نویز و هموارسازی' &&
                f.category !== 'لبه‌یابی و تیزسازی' &&
                f.category !== 'پردازش در حوزه فرکانس' &&
                f.category !== 'Additional' &&
                f.category !== 'Noise Filters'
            );
        } else {
            // For specific categories, use the mapped category
            filteredFilters = filters.filter(f => f.category === mappedCategory);
        }
    }

    filteredFilters.forEach(filter => {
        const card = document.createElement('div');
        card.className = 'filter-card bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/10 cursor-pointer hover:border-purple-400/50 transition-all duration-200';
        card.dataset.filterId = filter.id;

        if (filter.id === currentFilter) {
            card.classList.add('active', 'border-purple-400', 'bg-purple-500/20');
        }

        const icon = getFilterIcon(filter.category);

        // Use the description from the API response
        let displayName =  filter.name;

        card.innerHTML = `
            <div class="text-center mb-2">
                <i class="${icon} text-2xl text-purple-400"></i>
            </div>
            <h4 class="text-sm font-medium text-white text-center">${displayName}</h4>
        `;

        card.addEventListener('click', () => selectFilter(filter));
        grid.appendChild(card);
    });
}

// Select a filter
function selectFilter(filter) {
    currentFilter = filter.id;
    currentParams = {};

    // Update UI
    document.querySelectorAll('.filter-card').forEach(card => {
        card.classList.remove('active', 'border-purple-400', 'bg-purple-500/20');
    });
    document.querySelector(`[data-filter-id="${filter.id}"]`)?.classList.add('active', 'border-purple-400', 'bg-purple-500/20');

    // Update current filter info
    document.getElementById('currentFilterName').textContent = filter.name;
    document.getElementById('filterDescription').textContent = filter.description || filter.name || 'توضیحاتی موجود نیست';

    // Display formula if available
    const formulaDiv = document.createElement('div');
    formulaDiv.className = 'mt-3 p-3 bg-gray-700 rounded-lg text-center';
    if (filter.formula) {
        formulaDiv.innerHTML = '<div class="text-sm text-purple-300"><strong>فرمول:</strong><br>$' + filter.formula + '$</div>';
        // Render the MathJax formula after a short delay to ensure DOM insertion
        setTimeout(() => {
            if (typeof MathJax !== 'undefined' && MathJax.startup && MathJax.startup.document) {
                MathJax.typesetPromise([formulaDiv]).catch(err => console.log('MathJax error:', err.message));
            }
        }, 100);
    } else {
        formulaDiv.innerHTML = '<div class="text-sm text-gray-400">فرمول در دسترس نیست</div>';
    }

    // Show/hide settings
    if (filter.params && filter.params.length > 0) {
        renderFilterSettings(filter.params);
        // Add formula div after settings
        setTimeout(() => {
            document.getElementById('settingsContainer').appendChild(formulaDiv);
        }, 10);
    } else {
        document.getElementById('settingsContainer').innerHTML = `
            <div class="text-center text-gray-400 py-8">
                <i class="fas fa-sliders-h text-4xl mb-4"></i>
                <p>این فیلتر تنظیمات اضافی ندارد</p>
            </div>
        `;
        // Add formula div after settings
        setTimeout(() => {
            document.getElementById('settingsContainer').appendChild(formulaDiv);
        }, 10);
    }
}

// Render filter settings
function renderFilterSettings(params) {
    const container = document.getElementById('settingsContainer');
    container.innerHTML = '';

    params.forEach(param => {
        const paramValue = param.default || param.min;
        currentParams[param.name] = paramValue;

        // Determine if this parameter should be an integer
        const isIntegerParam = ['ksize', 'size', 'grid', 'bit', 'threshold', 'angle', 'n', 'D0'].some(intParam =>
            param.name.toLowerCase().includes(intParam)
        ) || param.step === 2; // Odd number steps like ksize typically need to be odd integers

        const paramDiv = document.createElement('div');
        paramDiv.className = 'bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10 mb-3';

        // Set step based on whether it should be an integer
        const stepValue = isIntegerParam ? 1 : (param.step || 0.1);

        paramDiv.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <label class="text-sm font-medium text-gray-300">${param.name}</label>
                <span class="text-sm font-bold text-purple-400" id="value-${param.name}">${paramValue}</span>
            </div>
            <input type="range"
                   id="param-${param.name}"
                   class="range-slider w-full"
                   min="${param.min}"
                   max="${param.max}"
                   value="${paramValue}"
                   step="${stepValue}"
                   onchange="updateParam('${param.name}', this.value)"
                   oninput="updateParam('${param.name}', this.value)">
            <div class="flex justify-between text-xs text-gray-400 mt-1">
                <span>${param.min}</span>
                <span>${param.max}</span>
            </div>
        `;
        container.appendChild(paramDiv);
    });
}

// Update parameter value
function updateParam(name, value) {
    // Determine if this parameter should be an integer
    const isIntegerParam = ['ksize', 'size', 'grid', 'bit', 'threshold', 'angle', 'n', 'D0'].some(intParam =>
        name.toLowerCase().includes(intParam)
    );

    // Round to integer if needed
    const processedValue = isIntegerParam ? Math.round(parseFloat(value)) : parseFloat(value);

    currentParams[name] = processedValue;
    document.getElementById(`value-${name}`).textContent = processedValue;
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showLoading(true);

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            uploadedFilename = data.filename;
            displayImages(file);
            showToast('تصویر با موفقیت آپلود شد', 'success');
        } else {
            showToast(data.error || 'خطا در آپلود تصویر', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('خطا در ارتباط با سرور', 'error');
    } finally {
        showLoading(false);
    }
}

// Display images
function displayImages(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const originalImg = document.getElementById('originalImage');
        const filteredImg = document.getElementById('filteredImage');

        originalImg.src = e.target.result;
        filteredImg.src = e.target.result;

        // Store the original image data for later use
        originalImageData = e.target.result;
        currentImageData = e.target.result; // Initially, current image is the original
        isUsingModifiedOriginal = false; // Reset the flag when a new image is loaded

        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('mainContent').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Apply filter
async function applyFilter() {
    if (!uploadedFilename && !originalImageData) {
        showToast('لطفاً ابتدا تصویری آپلود کنید', 'error');
        return;
    }

    showLoading(true);

    try {
        // Prepare the request payload
        const requestData = {
            filter: currentFilter,
            params: currentParams
        };

        // Apply filter to the appropriate image based on user's choice
        if (isUsingModifiedOriginal && currentImageData) {
            // If user has set a modified image as original, use current image data
            requestData.image_data = currentImageData;
        } else {
            // Otherwise, use the original image
            if (originalImageData) {
                requestData.image_data = originalImageData;
            } else {
                requestData.filename = uploadedFilename;
            }
        }

        const response = await fetch(`${API_URL}/apply-filter`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('filteredImage').src = data.image;
            // Update current image data to the newly filtered image
            currentImageData = data.image;
            showToast('فیلتر با موفقیت اعمال شد', 'success');
        } else {
            showToast(data.error || 'خطا در اعمال فیلتر', 'error');
        }
    } catch (error) {
        console.error('Apply filter error:', error);
        showToast('خطا در ارتباط با سرور', 'error');
    } finally {
        showLoading(false);
    }
}

// Download image
function downloadImage() {
    if (!currentImageData && !uploadedFilename) {
        showToast('هیچ تصویری برای دانلود وجود ندارد', 'error');
        return;
    }

    // If we have current image data, download that directly
    if (currentImageData) {
        const link = document.createElement('a');
        link.href = currentImageData;
        link.download = `filtered_image_${Date.now()}.png`;
        link.click();
        showToast('دانلود شروع شد', 'success');
    } else {
        // Fallback to server download endpoint
        const link = document.createElement('a');
        link.href = `${API_URL}/download/${currentFilter}/${uploadedFilename}`;
        link.download = `filtered_${uploadedFilename}`;
        link.click();
        showToast('دانلود شروع شد', 'success');
    }
}

// Set filtered image as original image for further processing
function setAsOriginal() {
    if (!uploadedFilename) {
        showToast('لطفاً ابتدا تصویری آپلود کنید', 'error');
        return;
    }

    // Get the current filtered image source
    const filteredImgSrc = document.getElementById('filteredImage').src;

    // Set the original image to be the same as the filtered image
    document.getElementById('originalImage').src = filteredImgSrc;

    // Also update the filtered image to maintain consistency
    document.getElementById('filteredImage').src = filteredImgSrc;

    // Update current image data to reflect the new "original"
    currentImageData = filteredImgSrc;

    // Set the flag to indicate we're now using a modified image as original
    isUsingModifiedOriginal = true;

    showToast('تصویر فیلتر شده به عنوان تصویر اصلی تنظیم شد', 'success');
}

// Reset to the original uploaded image
function resetToOriginal() {
    if (!originalImageData) {
        showToast('هیچ تصویر اصلی برای بازگشت وجود ندارد', 'error');
        return;
    }

    // Reset both images to the original uploaded image
    document.getElementById('originalImage').src = originalImageData;
    document.getElementById('filteredImage').src = originalImageData;

    // Reset current image data to original
    currentImageData = originalImageData;

    // Reset the flag to indicate we're back to using the original image
    isUsingModifiedOriginal = false;

    // Reset current filter to original
    currentFilter = 'original';
    currentParams = {};

    showToast('بازگشت به تصویر اصلی انجام شد', 'success');
}

// Reset image
async function resetImage() {
    if (uploadedFilename) {
        try {
            await fetch(`${API_URL}/cleanup/${uploadedFilename}`, {
                method: 'DELETE'
            });
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }

    uploadedFilename = null;
    originalImageData = null;
    currentImageData = null;
    isUsingModifiedOriginal = false; // Reset the flag
    currentFilter = 'original';
    currentParams = {};

    document.getElementById('fileInput').value = '';
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('mainContent').style.display = 'none';

    showToast('آماده برای تصویر جدید', 'success');
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('uploadBox').classList.add('border-purple-400', 'bg-purple-500/20');
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('uploadBox').classList.remove('border-purple-400', 'bg-purple-500/20');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('uploadBox').classList.remove('border-purple-400', 'bg-purple-500/20');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('fileInput').files = files;
        handleFileUpload({ target: { files } });
    }
}

// Initialize before/after slider
function initializeSlider() {
    const handle = document.getElementById('sliderHandle');
    const container = document.querySelector('.slider-container');
    const filteredImage = document.getElementById('filteredImage');

    let isDragging = false;

    handle.addEventListener('mousedown', startDrag);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', stopDrag);

    function startDrag(e) {
        isDragging = true;
        e.preventDefault();
    }

    function drag(e) {
        if (!isDragging) return;
        
        const rect = container.getBoundingClientRect();
        let x = e.clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        
        handle.style.left = x + 'px';
        filteredImage.style.clipPath = `polygon(0 0, ${x}px 0, ${x}px 100%, 0 100%)`;
    }

    function stopDrag() {
        isDragging = false;
    }
}

// Show loading
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toastIcon');
    const messageEl = document.getElementById('toastMessage');

    // Set icon and colors based on type
    if (type === 'success') {
        icon.className = 'fas fa-check-circle text-green-400 text-xl ml-3';
    } else if (type === 'error') {
        icon.className = 'fas fa-exclamation-circle text-red-400 text-xl ml-3';
    } else {
        icon.className = 'fas fa-info-circle text-yellow-400 text-xl ml-3';
    }

    messageEl.textContent = message;
    
    // Show toast
    toast.classList.remove('translate-x-full', 'opacity-0');
    toast.classList.add('translate-x-0', 'opacity-100');

    // Hide after 3 seconds
    setTimeout(() => {
        toast.classList.remove('translate-x-0', 'opacity-100');
        toast.classList.add('translate-x-full', 'opacity-0');
    }, 3000);
}

// Get filter icon based on category
function getFilterIcon(category) {
    const icons = {
        'اصلاح نور و کنتراست': 'fa-regular fa-sun',
        'پردازش هیستوگرام': 'fa-solid fa-chart-column',
        'کاهش نویز و هموارسازی': 'fa-solid fa-water',
        'لبه‌یابی و تیزسازی': 'fas fa-regular fa-burst',
        'پردازش در حوزه فرکانس': 'fas fa-wave-square',
        'Additional': 'fas fa-plus-circle',
        'Color': 'fa-solid fa-palette',
        'BW': 'fa-solid fa-circle-half-stroke',
        'Blur': 'fa-solid fa-water',
        'Edge Detection': 'fa-regular fa-draw-polygon',
        'Artistic': 'fa-regular fa-paintbrush',
        'Vintage': 'fa-solid fa-film',
        'Special': 'fa-solid fa-wand-sparkles',
        'Technical': 'fa-regular fa-star',
        'Distortion': 'fa-regular fa-expand',
        'Lighting': 'fa-regular fa-sun',
        'basic': 'fas fa-adjust',
        'blackwhite': 'fa-solid fa-circle-half-stroke',
        'edge': 'fa-regular fa-draw-polygon',
        'artistic': 'fa-regular fa-paintbrush',
        'vintage': 'fa-solid fa-film',
        'special': 'fa-solid fa-wand-sparkles',
        'enhancement': 'fa-regular fa-star',
        'distortion': 'fa-regular fa-expand',
        'light': 'fa-regular fa-sun',
        'Noise Filters': 'fa-solid fa-wave-square'
    };
    return icons[category] || 'fa-regular fa-image';
}

// Initialize MathJax when the page loads
document.addEventListener('DOMContentLoaded', function() {
    if (typeof MathJax !== 'undefined') {
        MathJax.startup.promise.then(() => {
            console.log('MathJax initialized');
        }).catch(err => console.log('MathJax error:', err.message));
    }
});