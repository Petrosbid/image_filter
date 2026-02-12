import cv2
import numpy as np
import math


class ComprehensiveImageProcessor:
    """
    پیاده‌سازی تمامی فیلترهای پردازش تصویر بر اساس کتاب گونزالس و فایل درسی.
    این کلاس شامل متادیتایی است که توضیحات و پارامترهای حیاتی را برای رابط کاربری برمی‌گرداند.
    """

    def __init__(self):
        # این دیکشنری نقش "کاتالوگ" را بازی می‌کند.
        # برای هر فیلتر: تابع اجرایی، توضیح فارسی، و پارامترهای قابل تنظیم تعریف شده است.
        self.filter_catalog = {
            # --- بخش ۱: اصلاح نور و کنتراست (Intensity Transformations) ---
            'negative': {
                'func': self.apply_negative,
                'category': 'اصلاح نور و کنتراست',
                'name': 'نگاتیو (Negative)',
                'description': 'معکوس کردن شدت روشنایی پیکسل‌ها (مشابه فیلم‌های عکاسی). مناسب برای تشخیص جزئیات سفید در پس‌زمینه تیره.',
                'formula': r's = L - 1 - r',
                'params': {}  # پارامتری ندارد
            },
            'log_transform': {
                'func': self.apply_log_transform,
                'category': 'اصلاح نور و کنتراست',
                'name': 'تبدیل لگاریتمی (Log Transform)',
                'description': 'گسترش دادن مقادیر تیره و فشرده کردن مقادیر روشن. برای دیدن جزئیات در نواحی بسیار تاریک استفاده می‌شود.',
                'formula': r's = c \cdot \log(1 + r)',
                'params': {
                    'c': 'ضریب مقیاس: افزایش آن باعث روشن‌تر شدن کلی تصویر خروجی می‌شود.'
                }
            },
            'gamma_correction': {
                'func': self.apply_gamma_correction,
                'category': 'اصلاح نور و کنتراست',
                'name': 'تصحیح گاما (Power-Law)',
                'description': 'انعطاف‌پذیرترین ابزار اصلاح نور. برخلاف لگاریتم، هم می‌تواند تصویر را روشن و هم تیره کند.',
                'formula': r's = c \cdot r^\gamma',
                'params': {
                    'gamma': 'مقدار گاما: اگر کمتر از 1 باشد (مثلاً 0.5) تصویر روشن می‌شود (اصلاح تصویر تاریک). اگر بیشتر از 1 باشد (مثلاً 2.5) تصویر تیره می‌شود (اصلاح تصویر شسته شده).',
                    'c': 'ضریب ثابت: تنظیم شدت کلی روشنایی.'
                }
            },
            'contrast_stretching': {
                'func': self.apply_contrast_stretching,
                'category': 'اصلاح نور و کنتراست',
                'name': 'بسط کنتراست (Contrast Stretching)',
                'description': 'کشیدن دامنه روشنایی تصویر برای پوشش کامل بازه ۰ تا ۲۵۵. کنتراست تصویر را با تیره کردن سایه‌ها و روشن کردن هایلایت‌ها افزایش می‌دهد.',
                'formula': r's = \begin{cases} s_1 \cdot \frac{r}{r_1} & r < r_1 \\ \frac{s_2-s_1}{r_2-r_1}(r-r_1)+s_1 & r_1 \leq r \leq r_2 \\ \frac{255-s_2}{255-r_2}(r-r_2)+s_2 & r > r_2 \end{cases}',
                'params': {
                    'r1': 'ورودی نقطه تیره (x1): نقاط تیره‌تر از این مقدار به سیاه نزدیک می‌شوند.',
                    's1': 'خروجی نقطه تیره (y1): معمولاً روی 0 تنظیم می‌شود.',
                    'r2': 'ورودی نقطه روشن (x2): نقاط روشن‌تر از این مقدار به سفید نزدیک می‌شوند.',
                    's2': 'خروجی نقطه روشن (y2): معمولاً روی 255 تنظیم می‌شود.'
                }
            },
            'gray_level_slicing': {
                'func': self.apply_gray_level_slicing,
                'category': 'اصلاح نور و کنتراست',
                'name': 'برش سطوح خاکستری (Gray-level Slicing)',
                'description': 'برجسته کردن یک بازه خاص از روشنایی (مثلاً بافت خاص یا رگ خونی) و حذف بقیه اطلاعات.',
                'formula': r's = \begin{cases} L & l \leq r \leq u \\ 0 \text{ or } r & \text{otherwise} \end{cases}',
                'params': {
                    'lower': 'حد پایین بازه: شروع روشنایی مورد نظر.',
                    'upper': 'حد بالا بازه: پایان روشنایی مورد نظر.',
                    'bg_mode': 'حالت پس‌زمینه: اگر فعال باشد، هر چیزی خارج از بازه سیاه می‌شود. اگر غیرفعال باشد، بقیه تصویر دست نخورده می‌ماند.'
                }
            },
            'bit_plane_slicing': {
                'func': self.apply_bit_plane_slicing,
                'category': 'اصلاح نور و کنتراست',
                'name': 'برش صفحات بیتی (Bit-plane Slicing)',
                'description': 'تجزیه تصویر به بیت‌های سازنده آن. بیت‌های بالا ساختار تصویر را دارند و بیت‌های پایین حاوی نویز هستند.',
                'formula': r's = (r \gg n) \& 1',
                'params': {
                    'bit': 'شماره بیت: از 0 تا 7. بیت 7 بیشترین اطلاعات بصری را دارد و بیت 0 شبیه به نویز تصادفی است.'
                }
            },

            # --- بخش ۲: پردازش هیستوگرام (Histogram Processing) ---
            'histogram_equalization': {
                'func': self.apply_histogram_equalization,
                'category': 'پردازش هیستوگرام',
                'name': 'همسان‌سازی هیستوگرام (HE)',
                'description': 'توزیع خودکار روشنایی‌ها برای دستیابی به بالاترین کنتراست سراسری. ممکن است نویز را در نواحی یکنواخت افزایش دهد.',
                'formula': r's_k = \sum_{j=0}^{k} \frac{n_j}{N} \cdot (L-1)',
                'params': {}
            },
            'clahe': {
                'func': self.apply_clahe,
                'category': 'پردازش هیستوگرام',
                'name': 'همسان‌سازی تطبیقی محدود (CLAHE)',
                'description': 'نسخه پیشرفته HE که روی پنجره‌های کوچک عمل می‌کند. جزئیات محلی را عالی نشان می‌دهد و از اشباع شدن نویز جلوگیری می‌کند.',
                'formula': r'\text{CLAHE: } s_k = \sum_{j=0}^{k} \frac{\min(h_j, \text{clip\_limit})}{N} \cdot (L-1)',
                'params': {
                    'clip_limit': 'حد برش: هرچه بیشتر باشد، کنتراست بیشتر می‌شود اما نویز هم بیشتر تقویت می‌شود (استاندارد: 2 تا 4).',
                    'grid_size': 'اندازه شبکه: ابعاد نواحی محلی (مثلاً 8). شبکه کوچکتر جزئیات ریزتر را تقویت می‌کند.'
                }
            },

            # --- بخش ۳: کاهش نویز و هموارسازی (Smoothing) ---
            'box_filter': {
                'func': self.apply_box_filter,
                'category': 'کاهش نویز و هموارسازی',
                'name': 'فیلتر میانگین (Box Filter)',
                'description': 'ساده‌ترین روش مات‌سازی با میانگین‌گیری. نویز را کم می‌کند اما لبه‌های تصویر را به شدت تار می‌کند.',
                'formula': r'g(x,y) = \frac{1}{M \cdot N} \sum_{i=-a}^{a} \sum_{j=-b}^{b} f(x+i, y+j)',
                'params': {
                    'ksize': 'اندازه پنجره: هرچه عدد بزرگتر باشد (مثلاً 9)، تصویر مات‌تر می‌شود و جزئیات بیشتری از بین می‌رود.'
                }
            },
            'gaussian_blur': {
                'func': self.apply_gaussian_blur,
                'category': 'کاهش نویز و هموارسازی',
                'name': 'فیلتر گوسی (Gaussian Blur)',
                'description': 'مات‌سازی نرم با وزن‌دهی گوسی. تاری طبیعی‌تری نسبت به فیلتر میانگین ایجاد می‌کند.',
                'formula': r'H(u,v) = e^{-\frac{u^2+v^2}{2\sigma^2}}',
                'params': {
                    'ksize': 'اندازه پنجره (فرد): محدوده اثرگذاری فیلتر.',
                    'sigma': 'انحراف معیار: پارامتر اصلی کنترل تاری. هرچه بیشتر باشد، تاری شدیدتر است (حتی با پنجره ثابت).'
                }
            },
            'bilateral_filter': {
                'func': self.apply_bilateral_filter,
                'category': 'کاهش نویز و هموارسازی',
                'name': 'فیلتر دوطرفه (Bilateral)',
                'description': 'پیشرفته‌ترین فیلتر هموارساز که نویز را حذف می‌کند اما لبه‌های تیز را حفظ می‌کند (مناسب رتوش پوست).',
                'formula': r'g(i,j) = \frac{1}{W_p} \sum_{k,l \in \Omega} f(k,l) \cdot w(i,j,k,l)',
                'params': {
                    'd': 'قطر همسایگی: اندازه ناحیه پیکسلی.',
                    'sigma_color': 'حساسیت رنگ: عدد بزرگتر یعنی رنگ‌های متفاوت‌تری با هم ترکیب شوند (کارتونی شدن).',
                    'sigma_space': 'حساسیت مکانی: عدد بزرگتر یعنی پیکسل‌های دورتر هم تاثیر بگذارند.'
                }
            },
            'median_filter': {
                'func': self.apply_median_filter,
                'category': 'کاهش نویز و هموارسازی',
                'name': 'فیلتر میانه (Median)',
                'description': 'فیلتر آماری غیرخطی. تنها راه موثر برای حذف نویزهای نمک و فلفلی (نقاط سفید و سیاه خالص) بدون مات کردن لبه‌ها.',
                'formula': r'g(x,y) = \text{median}\{f(x-i,y-j) | (i,j) \in S_{xy}\}',
                'params': {
                    'ksize': 'اندازه پنجره: باید فرد باشد. بزرگتر کردن آن نویزهای بزرگتر را حذف می‌کند اما تصویر را خمیری می‌کند.'
                }
            },
            'min_filter': {
                'func': self.apply_min_filter,
                'category': 'کاهش نویز و هموارسازی',
                'name': 'فیلتر مینیمم (Erosion)',
                'description': 'انتخاب تاریک‌ترین پیکسل در همسایگی. نقاط روشن (نویز نمک) را حذف می‌کند و اشیاء تیره را ضخیم‌تر می‌کند.',
                'formula': r'g(x,y) = \min\{f(x+s,y+t) | (s,t) \in S\}',
                'params': {
                    'ksize': 'قدرت فیلتر: هرچه بیشتر باشد، نواحی روشن بیشتر خورده می‌شوند.'
                }
            },
            'max_filter': {
                'func': self.apply_max_filter,
                'category': 'کاهش نویز و هموارسازی',
                'name': 'فیلتر ماکسیمم (Dilation)',
                'description': 'انتخاب روشن‌ترین پیکسل در همسایگی. نقاط تیره (نویز فلفل) را حذف می‌کند و اشیاء روشن را ضخیم‌تر می‌کند.',
                'formula': r'g(x,y) = \max\{f(x+s,y+t) | (s,t) \in S\}',
                'params': {
                    'ksize': 'قدرت فیلتر: هرچه بیشتر باشد، نواحی روشن بیشتر گسترش می‌یابند.'
                }
            },

            # --- بخش ۴: لبه‌یابی و تیزسازی (Sharpening & Edge Detection) ---
            'laplacian': {
                'func': self.apply_laplacian,
                'category': 'لبه‌یابی و تیزسازی',
                'name': 'لاپلاسین (Laplacian)',
                'description': 'استخراج لبه‌ها با مشتق دوم. به تغییرات ناگهانی در تمام جهات حساس است و خطوط دو لبه ایجاد می‌کند.',
                'formula': r'\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}',
                'params': {
                    'ksize': 'اندازه کرنل: حساسیت به ضخامت لبه.',
                    'scale': 'ضریب مقیاس: شدت خطوط لبه استخراج شده را تعیین می‌کند.'
                }
            },
            'unsharp_masking': {
                'func': self.apply_unsharp_masking,
                'category': 'لبه‌یابی و تیزسازی',
                'name': 'آنشارپ ماسکینگ (Unsharp Masking)',
                'description': 'روش استاندارد صنعتی برای شفاف‌سازی (Sharpening). با افزودن لبه‌ها به تصویر اصلی، تاری را از بین می‌برد.',
                'formula': r'g(x,y) = f(x,y) + k \cdot (f(x,y) - f_{smooth}(x,y))',
                'params': {
                    'amount': 'میزان تیز شدن: اگر 1 باشد استاندارد است، اگر بیشتر باشد (Highboost) اغراق‌آمیز می‌شود.',
                    'sigma': 'شعاع مات‌سازی: تعیین می‌کند چه جزئیاتی به عنوان "لبه" شناخته شوند.',
                    'threshold': 'آستانه: از تیز شدن نویزهای ریز جلوگیری می‌کند.'
                }
            },
            'sobel_edge': {
                'func': self.apply_sobel,
                'category': 'لبه‌یابی و تیزسازی',
                'name': 'لبه‌یاب سوبل (Sobel)',
                'description': 'لبه‌یابی بر اساس گرادیان (مشتق اول). لبه‌های قوی افقی یا عمودی را پیدا می‌کند و مقاومت خوبی در برابر نویز دارد.',
                'formula': r'G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}',
                'params': {
                    'ksize': 'اندازه کرنل: (1, 3, 5, 7).',
                    'direction': 'جهت: "x" برای لبه‌های عمودی، "y" برای افقی، "both" برای هر دو.'
                }
            },
            'canny_edge': {
                'func': self.apply_canny,
                'category': 'لبه‌یابی و تیزسازی',
                'name': 'لبه‌یاب کنی (Canny)',
                'description': 'بهینه‌ترین الگوریتم لبه‌یابی. خروجی آن خطوط نازک (تک پیکسل) و پیوسته است و نویز را حذف می‌کند.',
                'formula': r'\text{Canny: } \nabla f = \sqrt{G_x^2 + G_y^2}, \theta = \arctan(\frac{G_y}{G_x})',
                'params': {
                    'threshold1': 'آستانه پایین: لبه‌های ضعیف زیر این مقدار حذف می‌شوند.',
                    'threshold2': 'آستانه بالا: لبه‌های قوی بالای این مقدار حتماً نگه داشته می‌شوند.'
                }
            },

            # --- بخش ۵: پردازش در حوزه فرکانس (Frequency Domain) ---
            'ideal_lowpass': {
                'func': lambda img, **k: self.apply_frequency_filter(img, 'lowpass', 'ideal', **k),
                'category': 'پردازش در حوزه فرکانس',
                'name': 'پایین‌گذر ایده‌آل (Ideal Lowpass)',
                'description': 'حذف ناگهانی تمام فرکانس‌های بالا. باعث ایجاد اعوجاج "موج‌دار شدن" (Ringing) در لبه‌ها می‌شود.',
                'formula': r'H(u,v) = \begin{cases} 1 & D(u,v) \leq D_0 \\ 0 & D(u,v) > D_0 \end{cases}',
                'params': {
                    'D0': 'فرکانس قطع (شعاع): هرچه کمتر باشد، تصویر مات‌تر می‌شود و اثر موج‌دار بیشتر مشخص می‌شود.'
                }
            },
            'butterworth_lowpass': {
                'func': lambda img, **k: self.apply_frequency_filter(img, 'lowpass', 'butterworth', **k),
                'category': 'پردازش در حوزه فرکانس',
                'name': 'پایین‌گذر باتروارث (Butterworth LP)',
                'description': 'ایجاد تاری با لبه‌های نرم‌تر. انتخابی متعادل بین فیلتر ایده‌آل و گوسی است و اثر موج‌دار شدن را کنترل می‌کند.',
                'formula': r'H(u,v) = \frac{1}{1 + \left(\frac{D(u,v)}{D_0}\right)^{2n}}',
                'params': {
                    'D0': 'فرکانس قطع: میزان تاری.',
                    'n': 'مرتبه فیلتر: هرچه عدد بزرگتر باشد، رفتار فیلتر تیزتر شده و شبیه به "ایده‌آل" می‌شود (احتمال موج‌دار شدن).'
                }
            },
            'gaussian_lowpass': {
                'func': lambda img, **k: self.apply_frequency_filter(img, 'lowpass', 'gaussian', **k),
                'category': 'پردازش در حوزه فرکانس',
                'name': 'پایین‌گذر گوسی (Gaussian LP)',
                'description': 'نرم‌ترین حالت مات‌سازی بدون هیچ‌گونه اثر موج‌دار شدن (Ringing). در طبیعت و بینایی انسان کاربرد دارد.',
                'formula': r'H(u,v) = e^{-\frac{D^2(u,v)}{2D_0^2}}',
                'params': {
                    'D0': 'فرکانس قطع (سیگما): تعیین می‌کند چه میزان از جزئیات ریز حذف شوند.'
                }
            },
            'ideal_highpass': {
                'func': lambda img, **k: self.apply_frequency_filter(img, 'highpass', 'ideal', **k),
                'category': 'پردازش در حوزه فرکانس',
                'name': 'بالاگذر ایده‌آل (Ideal Highpass)',
                'description': 'حذف فرکانس‌های پایین و حفظ لبه‌ها. باعث ایجاد لبه‌های دوگانه و موج‌دار می‌شود.',
                'formula': r'H(u,v) = \begin{cases} 0 & D(u,v) \leq D_0 \\ 1 & D(u,v) > D_0 \end{cases}',
                'params': {
                    'D0': 'فرکانس قطع: هرچه کمتر باشد، جزئیات بیشتری از تصویر باقی می‌ماند.'
                }
            },
            'butterworth_highpass': {
                'func': lambda img, **k: self.apply_frequency_filter(img, 'highpass', 'butterworth', **k),
                'category': 'پردازش در حوزه فرکانس',
                'name': 'بالاگذر باتروارث (Butterworth HP)',
                'description': 'استخراج لبه‌ها با گذار نرم. لبه‌ها را تمیزتر از روش ایده‌آل نشان می‌دهد.',
                'formula': r'H(u,v) = \frac{1}{1 + \left(\frac{D_0}{D(u,v)}\right)^{2n}}',
                'params': {
                    'D0': 'فرکانس قطع: مرز بین بافت و لبه.',
                    'n': 'مرتبه فیلتر: کنترل شیب برش فرکانسی.'
                }
            },
            'gaussian_highpass': {
                'func': lambda img, **k: self.apply_frequency_filter(img, 'highpass', 'gaussian', **k),
                'category': 'پردازش در حوزه فرکانس',
                'name': 'بالاگذر گوسی (Gaussian HP)',
                'description': 'استخراج لبه بسیار نرم و طبیعی بدون نویزهای مصنوعی.',
                'formula': r'H(u,v) = 1 - e^{-\frac{D^2(u,v)}{2D_0^2}}',
                'params': {
                    'D0': 'فرکانس قطع: تعیین حساسیت فیلتر به جزئیات.'
                }
            },
            'fourier_spectrum': {
                'func': self.apply_fourier_spectrum,
                'category': 'پردازش حوزه فرکانس',
                'name': 'تبدیل فوریه (نمایش طیف)',
                'description': 'نمایش طیف دامنه (Magnitude Spectrum) تصویر. نقاط روشن مرکزی فرکانس‌های پایین و نقاط دورتر فرکانس‌های بالا را نشان می‌دهند.',
                'params': {}
            },
        }

    # --- متدهای کمکی ---
    def _ensure_gray(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _to_float(self, img):
        return img.astype(np.float32) / 255.0

    def _to_uint8(self, img):
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    # --- پیاده‌سازی فیلترها ---

    def apply_negative(self, img, **kwargs):
        # s = L - 1 - r
        return 255 - img

    def apply_log_transform(self, img, c=None, **kwargs):
        # s = c * log(1 + r)
        img_float = np.float32(img)
        if c is None:
            c = 255 / np.log(1 + np.max(img_float))
        log_image = c * (np.log(img_float + 1))
        return np.array(log_image, dtype=np.uint8)

    def apply_gamma_correction(self, img, gamma=1.0, c=1.0, **kwargs):
        # s = c * r ^ gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255 * c for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def apply_contrast_stretching(self, img, r1=70, s1=0, r2=140, s2=255, **kwargs):
        # تابع خطی تکه‌تکه
        img = self._ensure_gray(img)  # معمولا روی تک کانال اعمال می‌شود
        output = np.zeros_like(img)

        # بخش ۱: تیره
        idx1 = img < r1
        if r1 > 0:
            output[idx1] = (s1 / r1) * img[idx1]

        # بخش ۲: میانی (بسط داده شده)
        idx2 = (img >= r1) & (img <= r2)
        if (r2 - r1) > 0:
            output[idx2] = ((s2 - s1) / (r2 - r1)) * (img[idx2] - r1) + s1

        # بخش ۳: روشن
        idx3 = img > r2
        if (255 - r2) > 0:
            output[idx3] = ((255 - s2) / (255 - r2)) * (img[idx3] - r2) + s2

        return output

    def apply_gray_level_slicing(self, img, lower=100, upper=200, bg_mode=True, **kwargs):
        img = self._ensure_gray(img)
        rows, cols = img.shape
        output = img.copy()

        if bg_mode:
            # حالت ۱: بقیه نقاط سیاه شوند
            output = np.zeros((rows, cols), dtype=np.uint8)
            output[(img >= lower) & (img <= upper)] = 255
        else:
            # حالت ۲: بقیه نقاط دست‌نخورده بمانند (روشن کردن ناحیه هدف)
            output[(img >= lower) & (img <= upper)] = 255

        return output

    def apply_bit_plane_slicing(self, img, bit=7, **kwargs):
        img = self._ensure_gray(img)
        # اعمال ماسک بیتی: 1 << bit
        plane = img & (1 << bit)
        # نرمال‌سازی برای نمایش (اگر بیت ۱ است، سفید دیده شود)
        plane = plane * (255 >> bit)
        # یا روش ساده‌تر:
        # plane = (np.array([int(i) >> bit & 1 for i in img.flatten()]) * 255).reshape(img.shape)
        return plane

    def apply_histogram_equalization(self, img, **kwargs):
        if len(img.shape) == 3:
            # تبدیل به YUV برای همسان‌سازی فقط روی روشنایی
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return cv2.equalizeHist(img)

    def apply_clahe(self, img, clip_limit=2.0, grid_size=8, **kwargs):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return clahe.apply(img)

    def apply_box_filter(self, img, ksize=3, **kwargs):
        return cv2.blur(img, (ksize, ksize))

    def apply_gaussian_blur(self, img, ksize=5, sigma=1.0, **kwargs):
        # اگر ksize زوج باشد، +1 می‌شود
        if ksize % 2 == 0: ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def apply_median_filter(self, img, ksize=3, **kwargs):
        if ksize % 2 == 0: ksize += 1
        return cv2.medianBlur(img, ksize)

    def apply_min_filter(self, img, ksize=3, **kwargs):
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.erode(img, kernel)  # فرسایش (Erosion) همان فیلتر مینیمم است

    def apply_max_filter(self, img, ksize=3, **kwargs):
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.dilate(img, kernel)  # گسترش (Dilation) همان فیلتر ماکسیمم است

    def apply_bilateral_filter(self, img, d=9, sigma_color=75, sigma_space=75, **kwargs):
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def apply_laplacian(self, img, ksize=3, scale=1, **kwargs):
        # لاپلاسین معمولا خروجی منفی دارد، پس از CV_64F استفاده می‌کنیم
        if ksize not in [1, 3, 5, 7]: ksize = 3
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize, scale=scale)
        # تبدیل به absolute و سپس uint8
        lap = np.uint8(np.absolute(lap))
        return lap

    def apply_unsharp_masking(self, img, ksize=5, sigma=1.0, amount=1.5, threshold=0, **kwargs):
        blurred = self.apply_gaussian_blur(img, ksize, sigma)
        sharpened = float(amount + 1) * img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(img - blurred) < threshold
            np.copyto(sharpened, img, where=low_contrast_mask)
        return sharpened

    def apply_sobel(self, img, ksize=3, direction='both', **kwargs):
        img_gray = self._ensure_gray(img)
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)

        abs_sobelx = np.uint8(np.absolute(sobelx))
        abs_sobely = np.uint8(np.absolute(sobely))

        if direction == 'x':
            return abs_sobelx
        elif direction == 'y':
            return abs_sobely
        else:
            return cv2.bitwise_or(abs_sobelx, abs_sobely)

    def apply_canny(self, img, threshold1=100, threshold2=200, **kwargs):
        return cv2.Canny(img, threshold1, threshold2)

    # --- پیاده‌سازی فیلترهای حوزه فرکانس (پیچیده) ---
    def apply_frequency_filter(self, img, filter_type, shape, D0=30, n=2, **kwargs):
        """
        اجرای کامل پایپ‌لاین فیلترینگ در حوزه فرکانس:
        1. تبدیل فوریه 2. شیفت 3. ساخت ماسک 4. اعمال ماسک 5. معکوس فوریه
        """
        image_gray = self._ensure_gray(img)
        rows, cols = image_gray.shape
        crow, ccol = rows // 2, cols // 2

        # 1. FFT
        f = np.fft.fft2(image_gray)
        fshift = np.fft.fftshift(f)

        # 2. Create Mask (H)
        H = np.zeros((rows, cols), dtype=np.float32)
        # شبکه مختصات
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        # محاسبه فاصله D(u,v)
        D = np.sqrt((u - ccol) ** 2 + (v - crow) ** 2)

        if shape == 'ideal':
            if filter_type == 'lowpass':
                H[D <= D0] = 1
            else:  # highpass
                H[D > D0] = 1

        elif shape == 'gaussian':
            if filter_type == 'lowpass':
                H = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
            else:  # highpass
                H = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))

        elif shape == 'butterworth':
            # جلوگیری از تقسیم بر صفر
            epsilon = 1e-8
            if filter_type == 'lowpass':
                H = 1 / (1 + (D / (D0 + epsilon)) ** (2 * n))
            else:  # highpass
                H = 1 / (1 + ((D0 + epsilon) / (D + epsilon)) ** (2 * n))

        # 3. Apply Mask
        fshift_filtered = fshift * H

        # 4. Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # نرمال‌سازی برای نمایش
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        return img_back

    # --- رابط کاربری متنی ---
    def list_filters(self):
        """لیست کردن تمام فیلترها و پارامترهایشان"""
        print(f"{'نام فیلتر (Key)':<25} | {'دسته‌بندی':<20} | {'توضیح کوتاه'}")
        print("-" * 100)
        for key, info in self.filter_catalog.items():
            print(f"{key:<25} | {info['category']:<20} | {info['description']}")
            if info['params']:
                print(f"   پارامترهای حیاتی: {info['params']}")
            print("-" * 50)

    def run_filter(self, filter_name, image, **kwargs):
        """اجرای امن یک فیلتر با نام"""
        if filter_name in self.filter_catalog:
            func = self.filter_catalog[filter_name]['func']
            return func(image, **kwargs)
        else:
            raise ValueError(f"فیلتر '{filter_name}' یافت نشد.")

    def apply_fourier_spectrum(self, image):
        """
        اجرای تبدیل فوریه گسسته (DFT) و نمایش طیف دامنه.
        """
        # 1. تبدیل به خاکستری اگر تصویر رنگی باشد (فوریه معمولا روی تک کانال زده می‌شود)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. محاسبه تبدیل فوریه دو بعدی
        f = np.fft.fft2(gray)

        # 3. شیفت دادن مولفه صفر فرکانس (DC) به مرکز تصویر
        fshift = np.fft.fftshift(f)

        # 4. محاسبه طیف دامنه (Magnitude Spectrum)
        # استفاده از log برای فشرده‌سازی دامنه دینامیکی تا قابل مشاهده شود
        # عدد 1 اضافه می‌شود تا از log(0) جلوگیری شود
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        # 5. نرمال‌سازی تصویر خروجی بین 0 تا 255 برای نمایش
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

        return magnitude_spectrum.astype(np.uint8)
