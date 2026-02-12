import cv2
import numpy as np
import os


class ComprehensiveNoiseGenerator:
    """
    پیاده‌سازی انواع مدل‌های نویز در پردازش تصویر (بر اساس مفاهیم کتاب گونزالس).
    این کلاس شامل متادیتایی است که توضیحات و پارامترهای حیاتی را برای هر نویز برمی‌گرداند.
    """

    def __init__(self):
        # کاتالوگ نویزها: شامل تابع اجرایی، دسته‌بندی، نام فارسی و پارامترها
        self.noise_catalog = {
            # --- نویزهای رایج ---
            'gaussian': {
                'func': self.add_gaussian_noise,
                'category': 'مدل‌های احتمالاتی (PDF)',
                'name': 'نویز گوسی (Gaussian)',
                'description': 'افزودن مقادیر تصادفی با توزیع نرمال به تصویر. مدل‌سازی نویزهای الکترونیکی و حسگر.',
                'formula': r'f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}',
                'params': {'mean': 'میانگین (معمولا 0)', 'var': 'واریانس (شدت نویز)'}
            },
            'salt_and_pepper': {
                'func': self.add_salt_and_pepper_noise,
                'category': 'نویز ضربه‌ای',
                'name': 'نویز نمک و فلفل (Salt & Pepper)',
                'description': 'تغییر تصادفی برخی پیکسل‌ها به سفید (نمک) یا سیاه (فلفل).',
                'formula': r'g(x,y) = \begin{cases} 0 & \text{with probability } \frac{p \cdot a}{2} \\ 255 & \text{with probability } \frac{(1-p) \cdot a}{2} \\ f(x,y) & \text{with probability } 1-a \end{cases}',
                'params': {'s_vs_p': 'نسبت نمک به فلفل (0.5 یعنی برابر)', 'amount': 'چگالی کلی نویز (0 تا 1)'}
            },
            'speckle': {
                'func': self.add_speckle_noise,
                'category': 'نویز ضربی',
                'name': 'نویز خال‌دار (Speckle)',
                'description': 'نویز ضربی که متناسب با شدت روشنایی پیکسل است (I = I + I*Noise).',
                'formula': r'g(x,y) = f(x,y) + n(x,y) \cdot f(x,y)',
                'params': {'var': 'واریانس نویز'}
            },
            # --- سایر توزیع‌های آماری ---
            'uniform': {
                'func': self.add_uniform_noise,
                'category': 'مدل‌های احتمالاتی (PDF)',
                'name': 'نویز یکنواخت (Uniform)',
                'description': 'افزودن مقادیر تصادفی که احتمال وقوع آن‌ها در بازه [a, b] یکسان است.',
                'formula': r'f(x) = \begin{cases} \frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}',
                'params': {'low': 'حد پایین نویز', 'high': 'حد بالای نویز'}
            },
            'poisson': {
                'func': self.add_poisson_noise,
                'category': 'وابسته به سیگنال',
                'name': 'نویز پواسون (Poisson)',
                'description': 'نویزی که از طبیعت گسسته فوتون‌های نور ناشی می‌شود (نویز شات).',
                'formula': r'P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}',
                'params': {}  # معمولاً پارامتر خاصی ندارد و تابع شدت تصویر است
            },
            'rayleigh': {
                'func': self.add_rayleigh_noise,
                'category': 'مدل‌های احتمالاتی (PDF)',
                'name': 'نویز رایلی (Rayleigh)',
                'description': 'نویزی با توزیع نامتقارن، معمولاً در تصاویر رادار و فراصوت دیده می‌شود.',
                'formula': r'f(x) = \frac{x-a}{\sigma^2} e^{-\frac{(x-a)^2}{2\sigma^2}}, x \geq a',
                'params': {'a': 'جابجایی مکانی (a)', 'b': 'پارامتر مقیاس (b)'}
            },
            'exponential': {
                'func': self.add_exponential_noise,
                'category': 'مدل‌های احتمالاتی (PDF)',
                'name': 'نویز نمایی (Exponential)',
                'description': 'نویزی که از توزیع نمایی پیروی می‌کند.',
                'formula': r'f(x) = \lambda e^{-\lambda x}',
                'params': {'scale': 'مقیاس (1/lambda)'}
            },
            'erlang': {
                'func': self.add_erlang_noise,
                'category': 'مدل‌های احتمالاتی (PDF)',
                'name': 'نویز ارلانگ/گاما (Erlang/Gamma)',
                'description': 'توزیع گاما که اغلب در پردازش تصاویر لیزری و SAR کاربرد دارد.',
                'formula': r'f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}',
                'params': {'shape': 'پارامتر شکل (k یا b)', 'scale': 'پارامتر مقیاس (theta)'}
            }
        }

    # --- توابع کمکی ---
    def _normalize_and_clip(self, noisy_image):
        """تضمین اینکه مقادیر بین 0 تا 255 باشند و فرمت uint8 داشته باشند"""
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    # --- پیاده‌سازی نویزها ---

    def add_gaussian_noise(self, image, mean=0, var=0.01):
        # نرمال‌سازی تصویر به بازه 0-1 برای محاسبات دقیق‌تر
        image_float = image.astype(np.float32) / 255.0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = image_float + gauss
        return self._normalize_and_clip(noisy * 255.0)

    def add_salt_and_pepper_noise(self, image, s_vs_p=0.5, amount=0.05):
        row, col = image.shape[:2]
        ch = 1 if len(image.shape) == 2 else image.shape[2]

        out = np.copy(image)

        # افزودن نمک (سفید)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255

        # افزودن فلفل (سیاه)
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0

        return out

    def add_speckle_noise(self, image, var=0.04):
        image_float = image.astype(np.float32) / 255.0
        # نویز ضربی: Noise = Image + Image * Gaussian
        sigma = var ** 0.5
        gauss = np.random.normal(0, sigma, image.shape)
        noisy = image_float + image_float * gauss
        return self._normalize_and_clip(noisy * 255.0)

    def add_uniform_noise(self, image, low=-0.1, high=0.1):
        image_float = image.astype(np.float32) / 255.0
        uniform_noise = np.random.uniform(low, high, image.shape)
        noisy = image_float + uniform_noise
        return self._normalize_and_clip(noisy * 255.0)

    def add_poisson_noise(self, image):
        # نویز پواسون بر اساس تعداد رویدادها (فوتون‌ها) کار می‌کند
        # مقادیر باید متناسب با شدت باشند
        image_float = image.astype(np.float32)
        # برای جلوگیری از خطا در مقادیر منفی (هرچند تصویر منفی نیست)
        vals = len(np.unique(image_float))
        vals = 2 ** np.ceil(np.log2(vals))

        # شبیه‌سازی: اعمال توزیع پواسون روی تصویر مقیاس‌دهی شده
        noisy = np.random.poisson(image_float)
        return self._normalize_and_clip(noisy)

    def add_rayleigh_noise(self, image, a=0, b=0.1):
        # فرمول رایلی: P(z) = (2/b)*(z-a)*exp(-(z-a)^2/b) for z >= a
        # در numpy.random.rayleigh پارامتر scale همان mode است (sigma)
        # واریانس رایلی برابر است با (4 - pi)/2 * scale^2
        # اینجا ما یک نویز افزودنی رایلی ایجاد می‌کنیم
        image_float = image.astype(np.float32) / 255.0

        # پارامتر scale برای تابع numpy
        scale = np.sqrt(b / 2) if b > 0 else 0.1
        noise = np.random.rayleigh(scale, image.shape)

        # اضافه کردن آفست a (شیفت دادن هیستوگرام نویز)
        noisy = image_float + (noise + a)
        return self._normalize_and_clip(noisy * 255.0)

    def add_exponential_noise(self, image, scale=0.1):
        # تابع نمایی: P(z) = a * exp(-a*z)
        # در numpy: scale = 1/a (mean)
        image_float = image.astype(np.float32) / 255.0
        noise = np.random.exponential(scale, image.shape)
        noisy = image_float + noise
        return self._normalize_and_clip(noisy * 255.0)

    def add_erlang_noise(self, image, shape=2.0, scale=0.1):
        # نویز ارلانگ مورد خاصی از توزیع گاما است که shape عدد صحیح است
        image_float = image.astype(np.float32) / 255.0
        noise = np.random.gamma(shape, scale, image.shape)
        noisy = image_float + noise
        return self._normalize_and_clip(noisy * 255.0)

    # --- رابط کاربری متنی ---
    def list_noises(self):
        """لیست کردن تمام نویزها و پارامترهایشان"""
        print(f"{'نام نویز (Key)':<25} | {'دسته‌بندی':<25} | {'توضیح کوتاه'}")
        print("-" * 110)
        for key, info in self.noise_catalog.items():
            print(f"{key:<25} | {info['category']:<25} | {info['description']}")
            if info['params']:
                print(f"   پارامترها: {info['params']}")
            print("-" * 50)

    def run_noise(self, noise_name, image, **kwargs):
        """اجرای امن یک نویز با نام مشخص"""
        if noise_name in self.noise_catalog:
            print(f">> در حال اعمال {self.noise_catalog[noise_name]['name']}...")
            func = self.noise_catalog[noise_name]['func']
            return func(image, **kwargs)
        else:
            raise ValueError(f"نویز '{noise_name}' یافت نشد.")