from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from Additional_filters import ImageFilters
from realtime_filters import RealtimeFilterProcessor, REALTIME_OPTIMIZED_FILTERS, HEAVY_FILTERS
import uuid
from main_filters import  ComprehensiveImageProcessor
from main_Noises import ComprehensiveNoiseGenerator

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10*1024*1024)

# تنظیمات
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ایجاد پوشه آپلود
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ایجاد instance از فیلترها
image_filters = ImageFilters()
comprehensive_processor = ComprehensiveImageProcessor()
realtime_processor = RealtimeFilterProcessor()
noise_generator = ComprehensiveNoiseGenerator()

def allowed_file(filename):
    """بررسی فرمت فایل"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array):
    """تبدیل آرایه numpy به base64"""
    pil_img = Image.fromarray(image_array.astype('uint8'))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# WebSocket Events for Real-time filtering
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to real-time filter server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('set_filter')
def handle_set_filter(data):
    """تنظیم فیلتر برای پردازش real-time"""
    filter_name = data.get('filter', 'original')
    params = data.get('params', {})
    
    print(f"Setting filter: {filter_name} with params: {params}")  # Debug
    
    realtime_processor.set_filter(filter_name, params)
    emit('filter_set', {'filter': filter_name, 'status': 'success'})

@socketio.on('process_frame')
def handle_process_frame(data):
    """پردازش یک فریم ویدیو"""
    try:
        frame_data = data.get('frame')
        if frame_data:
            processed_frame = realtime_processor.process_frame(frame_data)
            if processed_frame:
                emit('processed_frame', {'frame': processed_frame})
            else:
                # اگر فریم پردازش نشد، فریم اصلی را برگردان
                emit('processed_frame', {'frame': frame_data})
    except Exception as e:
        print(f"Error in handle_process_frame: {e}")
        # در صورت خطا، فریم اصلی را برگردان
        emit('processed_frame', {'frame': data.get('frame')})

@app.route('/api/realtime/filters', methods=['GET'])
def get_realtime_filters():
    """دریافت لیست فیلترهای مناسب برای real-time"""
    filters = []
    all_filters = get_filters().json['filters']

    for f in all_filters:
        filter_copy = f.copy()  # Create a copy to avoid modifying the original
        if f['id'] in REALTIME_OPTIMIZED_FILTERS:
            filter_copy['realtime_safe'] = True
            filter_copy['performance'] = 'optimal'
            filters.append(filter_copy)
        elif f['id'] in HEAVY_FILTERS:
            filter_copy['realtime_safe'] = False
            filter_copy['performance'] = 'heavy'
            filter_copy['performance_warning'] = 'ممکن است کندی ایجاد کند'
            filters.append(filter_copy)
        else:
            # فیلترهای متوسط
            filter_copy['realtime_safe'] = True
            filter_copy['performance'] = 'medium'
            filters.append(filter_copy)

    return jsonify({'filters': filters})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """آپلود تصویر"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'فایلی انتخاب نشده است'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'فایلی انتخاب نشده است'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'فرمت فایل پشتیبانی نمی‌شود'}), 400
        
        # بررسی حجم فایل
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        if file_length > MAX_FILE_SIZE:
            return jsonify({'error': 'حجم فایل بیش از 10 مگابایت است'}), 400
        file.seek(0)
        
        # ذخیره فایل با نام یکتا
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # خواندن تصویر و ارسال اطلاعات اولیه
        img = cv2.imread(filepath)
        height, width = img.shape[:2]
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'original_name': filename,
            'width': width,
            'height': height
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/apply-filter', methods=['POST'])
def apply_filter():
    """اعمال فیلتر روی تصویر"""
    try:
        data = request.json
        filename = data.get('filename')
        image_data = data.get('image_data')  # New parameter to accept image data directly
        filter_name = data.get('filter')
        params = data.get('params', {})

        if not (filename or image_data) or not filter_name:
            return jsonify({'error': 'پارامترهای ناقص'}), 400

        # Determine source of image
        if image_data:
            # Decode base64 image data
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': 'فرمت تصویر نامعتبر است'}), 400
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif filename:
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            if not os.path.exists(filepath):
                return jsonify({'error': 'فایل یافت نشد'}), 404

            # خواندن تصویر
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'نمی‌توان تصویر را خواند'}), 400
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return jsonify({'error': 'هیچ منبع تصویری مشخص نشده است'}), 400

        # تعیین اینکه از کدام کلاس فیلتر استفاده شود
        if filter_name in comprehensive_processor.filter_catalog:
            # استفاده از ComprehensiveImageProcessor
            try:
                filtered_image = comprehensive_processor.run_filter(filter_name, img_rgb, **params)
            except Exception as e:
                return jsonify({'error': f'خطا در اعمال فیلتر "{filter_name}": {str(e)}'}), 500
        elif filter_name in image_filters.filter_catalog:
            # استفاده از ImageFilters
            try:
                filtered_image = image_filters.run_filter(filter_name, img_rgb, **params)
            except Exception as e:
                return jsonify({'error': f'خطا در اعمال فیلتر "{filter_name}": {str(e)}'}), 500
        elif filter_name in noise_generator.noise_catalog:
            # استفاده از ComprehensiveNoiseGenerator
            try:
                filtered_image = noise_generator.run_noise(filter_name, img_rgb, **params)
            except Exception as e:
                return jsonify({'error': f'خطا در اعمال فیلتر نویز "{filter_name}": {str(e)}'}), 500
        else:
            return jsonify({'error': f'فیلتر "{filter_name}" یافت نشد'}), 404

        # تبدیل به base64
        image_base64 = image_to_base64(filtered_image)

        return jsonify({
            'success': True,
            'image': image_base64,
            'filter': filter_name
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'خطا در اعمال فیلتر: {str(e)}'}), 500

@app.route('/api/get-filters', methods=['GET'])
def get_filters():
    """دریافت لیست فیلترهای موجود"""
    filters = []

    # اضافه کردن فیلترهای از ComprehensiveImageProcessor
    for filter_id, filter_info in comprehensive_processor.filter_catalog.items():
        # Check if filter_info is a dictionary
        if not isinstance(filter_info, dict):
            print(f"Warning: filter_info for {filter_id} is not a dictionary: {type(filter_info)}")
            continue

        filter_dict = {
            'id': filter_id,
            'name': filter_info.get('name', filter_id),  # This is the display name, fallback to id
            'description': filter_info.get('description', ''),  # This is the detailed description, fallback to empty
            'category': filter_info.get('category', 'Uncategorized'),  # حفظ نام دسته بندی اصلی, fallback to Uncategorized
        }

        # اضافه کردن پارامترها اگر وجود داشته باشند
        params = filter_info.get('params', {})
        if params:
            param_list = []
            for param_name, param_desc in params.items():
                # تعیین مقدار پیش‌فرض بر اساس نام پارامتر
                if 'factor' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 0.1, 'max': 3.0, 'default': 1.0})
                elif 'gamma' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 0.1, 'max': 3.0, 'default': 1.0})
                elif 'threshold' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 0, 'max': 255, 'default': 128})
                elif 'clip' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 0.5, 'max': 5.0, 'default': 2.0})
                elif 'grid' in param_name.lower() or 'size' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 2, 'max': 16, 'default': 8})
                elif 'ksize' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 3, 'max': 31, 'default': 5, 'step': 2})
                elif 'sigma' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 0.1, 'max': 10.0, 'default': 1.0})
                elif 'angle' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 0, 'max': 360, 'default': 0})
                elif 'shift' in param_name.lower():
                    param_list.append({'name': param_name, 'min': -90, 'max': 90, 'default': 0})
                elif 'bit' in param_name.lower():
                    param_list.append({'name': param_name, 'min': 0, 'max': 7, 'default': 7})
                elif 'D0' in param_name.upper():
                    param_list.append({'name': param_name, 'min': 10, 'max': 100, 'default': 30})
                elif 'n' in param_name.lower():  # مرتبه فیلتر Butterworth
                    param_list.append({'name': param_name, 'min': 1, 'max': 10, 'default': 2})
                else:
                    # پارامتر عمومی
                    param_list.append({'name': param_name, 'min': 0, 'max': 100, 'default': 50})

            filter_dict['params'] = param_list

        # اضافه کردن فرمول اگر وجود داشته باشد
        formula = filter_info.get('formula', '')
        if formula:
            filter_dict['formula'] = formula

        filters.append(filter_dict)

    # اضافه کردن فیلترهای از ImageFilters
    for filter_id, filter_info in image_filters.filter_catalog.items():
        # Check if filter_info is a dictionary
        if not isinstance(filter_info, dict):
            print(f"Warning: filter_info for {filter_id} is not a dictionary: {type(filter_info)}")
            continue

        # اگر فیلتر قبلا اضافه شده بود، اضافه نکن
        if not any(f['id'] == filter_id for f in filters):
            filter_dict = {
                'id': filter_id,
                'name': filter_info.get('name', filter_id),  # This is the display name, fallback to id
                'description': filter_info.get('description', ''),  # This is the detailed description, fallback to empty
                'category': filter_info.get('category', 'Uncategorized'),  # حفظ نام دسته بندی اصلی, fallback to Uncategorized
            }

            # اضافه کردن پارامترها اگر وجود داشته باشند
            params = filter_info.get('params', {})
            if params:
                param_list = []
                for param_name, param_desc in params.items():
                    # تعیین مقدار پیش‌فرض بر اساس نام پارامتر
                    if 'factor' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.1, 'max': 3.0, 'default': 1.0})
                    elif 'shift' in param_name.lower():
                        param_list.append({'name': param_name, 'min': -90, 'max': 90, 'default': 30})
                    elif 'amount' in param_name.lower():
                        param_list.append({'name': param_name, 'min': -100, 'max': 100, 'default': 50})
                    elif 'color' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0, 'max': 255, 'default': 128})
                    elif 'red' in param_name.lower() or 'green' in param_name.lower() or 'blue' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.5, 'max': 1.5, 'default': 1.0})
                    else:
                        # پارامتر عمومی
                        param_list.append({'name': param_name, 'min': 0, 'max': 100, 'default': 50})

                filter_dict['params'] = param_list

            # اضافه کردن فرمول اگر وجود داشته باشد
            formula = filter_info.get('formula', '')
            if formula:
                filter_dict['formula'] = formula

            filters.append(filter_dict)

    # اضافه کردن فیلترهای نویز از ComprehensiveNoiseGenerator
    for filter_id, filter_info in noise_generator.noise_catalog.items():
        # Check if filter_info is a dictionary
        if not isinstance(filter_info, dict):
            print(f"Warning: filter_info for {filter_id} is not a dictionary: {type(filter_info)}")
            continue

        # اگر فیلتر قبلا اضافه شده بود، اضافه نکن
        if not any(f['id'] == filter_id for f in filters):
            filter_dict = {
                'id': filter_id,
                'name': filter_info.get('name', filter_id),  # This is the display name, fallback to id
                'description': filter_info.get('description', ''),  # This is the detailed description, fallback to empty
                'category': 'Noise Filters',  # حفظ نام دسته بندی اصلی, fallback to Uncategorized
            }

            # اضافه کردن پارامترها اگر وجود داشته باشند
            params = filter_info.get('params', {})
            if params:
                param_list = []
                for param_name, param_desc in params.items():
                    # تعیین مقدار پیش‌فرض بر اساس نام پارامتر
                    if 'mean' in param_name.lower():
                        param_list.append({'name': param_name, 'min': -0.5, 'max': 0.5, 'default': 0.0})
                    elif 'var' in param_name.lower() or 'variance' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.0, 'max': 0.1, 'default': 0.01})
                    elif 'amount' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.0, 'max': 1.0, 'default': 0.05})
                    elif 's_vs_p' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.0, 'max': 1.0, 'default': 0.5})
                    elif 'low' in param_name.lower():
                        param_list.append({'name': param_name, 'min': -0.5, 'max': 0.0, 'default': -0.1})
                    elif 'high' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.0, 'max': 0.5, 'default': 0.1})
                    elif 'scale' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.01, 'max': 0.5, 'default': 0.1})
                    elif 'a' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.0, 'max': 1.0, 'default': 0.0})
                    elif 'b' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 0.01, 'max': 1.0, 'default': 0.1})
                    elif 'shape' in param_name.lower():
                        param_list.append({'name': param_name, 'min': 1.0, 'max': 10.0, 'default': 2.0})
                    else:
                        # پارامتر عمومی
                        param_list.append({'name': param_name, 'min': 0, 'max': 1, 'default': 0.5})

                filter_dict['params'] = param_list

            # اضافه کردن فرمول اگر وجود داشته باشد
            formula = filter_info.get('formula', '')
            if formula:
                filter_dict['formula'] = formula

            filters.append(filter_dict)

    # اضافه کردن فیلترهای قدیمی برای سازگاری
    legacy_filters = [
        # فیلترهای پایه - بهینه برای real-time
        {'id': 'grayscale', 'name': 'سیاه و سفید', 'category': 'Additional'},
        {'id': 'brightness', 'name': 'روشنایی', 'category': 'Additional', 'params': [{'name': 'factor', 'min': 0.5, 'max': 2.0, 'default': 1.5}]},
        {'id': 'contrast', 'name': 'کنتراست', 'category': 'Additional', 'params': [{'name': 'factor', 'min': 0.5, 'max': 2.0, 'default': 1.5}]},
        {'id': 'gaussian_blur', 'name': 'محو گاوسی', 'category': 'Additional', 'params': [{'name': 'kernel_size', 'min': 3, 'max': 31, 'default': 15, 'step': 2}]},
        {'id': 'motion_blur', 'name': 'محو حرکتی', 'category': 'Additional', 'params': [
            {'name': 'size', 'min': 5, 'max': 25, 'default': 15, 'step': 2},
            {'name': 'angle', 'min': 0, 'max': 360, 'default': 0}
        ]},
        {'id': 'denoise', 'name': 'حذف نویز', 'category': 'Additional', 'params': [{'name': 'h', 'min': 5, 'max': 20, 'default': 10}]},
    ]

    # اضافه کردن فیلترهای قدیمی که در کاتالوگ‌های جدید نیستند
    for legacy_filter in legacy_filters:
        if not any(f['id'] == legacy_filter['id'] for f in filters):
            filters.append(legacy_filter)

    return jsonify({'filters': filters})

@app.route('/api/download/<filter_name>/<filename>', methods=['GET'])
def download_image(filter_name, filename):
    """دانلود تصویر فیلتر شده"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'فایل یافت نشد'}), 404
        
        # اعمال فیلتر
        filtered_image = image_filters.apply_filter(filepath, filter_name)
        
        # ذخیره در حافظه
        pil_img = Image.fromarray(filtered_image.astype('uint8'))
        img_io = io.BytesIO()
        pil_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png', 
                        download_name=f'{filter_name}_{filename}')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<filename>', methods=['DELETE'])
def cleanup(filename):
    """حذف فایل آپلود شده"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')