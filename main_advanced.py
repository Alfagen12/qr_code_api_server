"""
Улучшенный QR Code Scanner API Server
Использует продвинутые методы обработки изображений для повышения точности распознавания
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import os
import uvicorn
from typing import Dict, Any, List
import cv2
import numpy as np
from skimage import filters, morphology, exposure, measure
from skimage.color import rgb2gray
from skimage.transform import rotate
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Попытка импорта pyzbar
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
    logger.info("pyzbar доступен")
except ImportError:
    PYZBAR_AVAILABLE = False
    logger.warning("pyzbar недоступен, используется только OpenCV")

app = FastAPI(
    title="Advanced QR Code Scanner API",
    description="Продвинутый API для распознавания QR кодов с улучшенной обработкой изображений",
    version="2.0.0"
)

class AdvancedQRProcessor:
    """Класс для продвинутой обработки и распознавания QR кодов"""
    
    @staticmethod
    def preprocess_image(image: Image.Image) -> List[np.ndarray]:
        """
        Создает несколько вариантов предобработанного изображения
        для повышения вероятности успешного распознавания
        """
        # Конвертируем в numpy array
        img_array = np.array(image)
        
        processed_images = []
        
        # 1. Оригинальное изображение
        if len(img_array.shape) == 3:
            gray_original = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_original = img_array.copy()
        processed_images.append(gray_original)
        
        # Масштабирование для больших изображений
        height, width = gray_original.shape
        if width > 1000 or height > 1000:
            # Создаем уменьшенную версию для лучшего распознавания
            scale_factor = min(800/width, 800/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized = cv2.resize(gray_original, (new_width, new_height), interpolation=cv2.INTER_AREA)
            processed_images.append(resized)
            logger.info(f"Добавлено уменьшенное изображение: {new_width}x{new_height}")
        
        # 2. Улучшение контраста (CLAHE) с разными параметрами
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced1 = clahe1.apply(gray_original)
        processed_images.append(enhanced1)
        
        clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
        enhanced2 = clahe2.apply(gray_original)
        processed_images.append(enhanced2)
        
        # 3. Гауссово размытие + повышение резкости
        blurred = cv2.GaussianBlur(gray_original, (3, 3), 0)
        sharpened = cv2.addWeighted(gray_original, 1.5, blurred, -0.5, 0)
        processed_images.append(sharpened)
        
        # 4. Сильное повышение резкости для размытых изображений
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_strong = cv2.filter2D(gray_original, -1, kernel_sharp)
        processed_images.append(sharpened_strong)
        
        # 5. Бинаризация (Otsu)
        _, binary_otsu = cv2.threshold(gray_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(binary_otsu)
        
        # 6. Адаптивная бинаризация с разными параметрами
        adaptive_thresh1 = cv2.adaptiveThreshold(
            gray_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(adaptive_thresh1)
        
        adaptive_thresh2 = cv2.adaptiveThreshold(
            gray_original, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )
        processed_images.append(adaptive_thresh2)
        
        # 7. Морфологические операции
        kernel = np.ones((2,2), np.uint8)
        morphed = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morphed)
        
        # 8. Удаление шума
        denoised = cv2.medianBlur(gray_original, 3)
        processed_images.append(denoised)
        
        # 9. Билатеральный фильтр для сохранения краев
        bilateral = cv2.bilateralFilter(gray_original, 9, 75, 75)
        processed_images.append(bilateral)
        
        # 10. Эквализация гистограммы
        equalized = cv2.equalizeHist(gray_original)
        processed_images.append(equalized)
        
        # 11. Усиление контраста через конвертацию
        contrast_enhanced = cv2.convertScaleAbs(gray_original, alpha=1.3, beta=10)
        processed_images.append(contrast_enhanced)
        
        # 12. Инверсия (для светлых QR на темном фоне)
        inverted = cv2.bitwise_not(gray_original)
        processed_images.append(inverted)
        
        return processed_images
    
    @staticmethod
    def enhance_with_pil(image: Image.Image) -> List[Image.Image]:
        """
        Улучшение изображения с помощью PIL
        """
        enhanced_images = [image]
        
        # Увеличение контраста
        enhancer = ImageEnhance.Contrast(image)
        enhanced_images.append(enhancer.enhance(1.5))
        enhanced_images.append(enhancer.enhance(2.0))
        
        # Увеличение резкости
        enhancer = ImageEnhance.Sharpness(image)
        enhanced_images.append(enhancer.enhance(1.5))
        enhanced_images.append(enhancer.enhance(2.0))
        
        # Увеличение яркости
        enhancer = ImageEnhance.Brightness(image)
        enhanced_images.append(enhancer.enhance(1.2))
        enhanced_images.append(enhancer.enhance(0.8))
        
        return enhanced_images
    
    @staticmethod
    def rotate_image_variants(image: np.ndarray) -> List[np.ndarray]:
        """
        Создает варианты изображения с поворотами
        """
        rotated_images = [image]
        
        # Повороты на небольшие углы
        for angle in [-5, -2, 2, 5, -10, 10]:
            rows, cols = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rotated_images.append(rotated)
        
        return rotated_images
    
    @staticmethod
    def detect_with_opencv(image: np.ndarray) -> List[Dict]:
        """
        Распознавание QR кода с помощью OpenCV
        """
        results = []
        
        try:
            # Метод 1: Стандартный детектор
            detector = cv2.QRCodeDetector()
            data, vertices_array, binary_qrcode = detector.detectAndDecode(image)
            
            if data:
                results.append({
                    'data': data,
                    'type': 'QRCODE',
                    'confidence': 1.0,
                    'method': 'opencv_standard'
                })
            
            # Метод 2: Детектор с настройками для сложных случаев
            try:
                # Попробуем найти QR контуры вручную
                detector_improved = cv2.QRCodeDetector()
                # Уменьшаем требования к точности
                detector_improved.setEpsX(0.4)
                detector_improved.setEpsY(0.4)
                
                data_improved, vertices_improved, _ = detector_improved.detectAndDecode(image)
                if data_improved and data_improved != data:
                    results.append({
                        'data': data_improved,
                        'type': 'QRCODE', 
                        'confidence': 0.9,
                        'method': 'opencv_improved'
                    })
            except:
                pass
            
            # Метод 3: Попробуем с различными уровнями размытия
            if not data:
                for blur_size in [1, 3, 5]:
                    try:
                        blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
                        data_blur, _, _ = detector.detectAndDecode(blurred)
                        if data_blur:
                            results.append({
                                'data': data_blur,
                                'type': 'QRCODE',
                                'confidence': 0.8,
                                'method': f'opencv_blur_{blur_size}'
                            })
                            break
                    except:
                        continue
            
            # Метод 4: Попробуем с масштабированием
            if not data:
                height, width = image.shape
                for scale in [0.5, 1.5, 2.0]:
                    try:
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        if new_width > 50 and new_height > 50 and new_width < 4000 and new_height < 4000:
                            scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                            data_scaled, _, _ = detector.detectAndDecode(scaled)
                            if data_scaled:
                                results.append({
                                    'data': data_scaled,
                                    'type': 'QRCODE',
                                    'confidence': 0.7,
                                    'method': f'opencv_scale_{scale}'
                                })
                                break
                    except:
                        continue
                
        except Exception as e:
            logger.debug(f"OpenCV detection failed: {e}")
        
        return results
    
    @staticmethod
    def detect_with_pyzbar(image) -> List[Dict]:
        """
        Распознавание QR кода с помощью pyzbar
        """
        results = []
        
        if not PYZBAR_AVAILABLE:
            return results
            
        try:
            # Если image - это numpy array, конвертируем в PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            qr_codes = pyzbar.decode(pil_image)
            
            for qr_code in qr_codes:
                results.append({
                    'data': qr_code.data.decode('utf-8'),
                    'type': qr_code.type,
                    'confidence': 1.0,
                    'method': 'pyzbar',
                    'rect': {
                        'left': qr_code.rect.left,
                        'top': qr_code.rect.top, 
                        'width': qr_code.rect.width,
                        'height': qr_code.rect.height
                    }
                })
                
        except Exception as e:
            logger.debug(f"pyzbar detection failed: {e}")
            
        return results
    
    @classmethod
    def scan_qr_comprehensive(cls, image: Image.Image) -> List[Dict]:
        """
        Комплексное сканирование QR кода с использованием всех доступных методов
        """
        all_results = []
        width, height = image.size
        
        logger.info(f"Анализируем изображение размером {width}x{height}")
        
        # Для больших изображений попробуем также поиск по частям
        if width > 800 or height > 800:
            logger.info("Большое изображение - добавляем поиск по секциям")
            
            # Разделим изображение на перекрывающиеся части
            for y_offset in [0, height//3, height//2]:
                for x_offset in [0, width//3, width//2]:
                    if y_offset + 400 <= height and x_offset + 400 <= width:
                        # Вырезаем фрагмент
                        crop_box = (x_offset, y_offset, x_offset + 400, y_offset + 400)
                        cropped = image.crop(crop_box)
                        
                        # Быстрая проверка на pyzbar
                        if PYZBAR_AVAILABLE:
                            try:
                                qr_codes = pyzbar.decode(cropped)
                                for qr in qr_codes:
                                    all_results.append({
                                        'data': qr.data.decode('utf-8'),
                                        'type': qr.type,
                                        'confidence': 1.0,
                                        'method': 'pyzbar_crop',
                                        'preprocessing': f'crop_{x_offset}x{y_offset}'
                                    })
                            except:
                                pass
        
        # 1. Попробуем с оригинальным изображением
        logger.info("Сканирование оригинального изображения...")
        
        # PIL варианты
        pil_variants = cls.enhance_with_pil(image)
        for i, pil_variant in enumerate(pil_variants):
            results = cls.detect_with_pyzbar(pil_variant)
            for result in results:
                result['preprocessing'] = f'pil_variant_{i}'
                all_results.append(result)
        
        # 2. Предобработанные варианты
        logger.info("Сканирование предобработанных изображений...")
        processed_images = cls.preprocess_image(image)
        
        for i, processed_img in enumerate(processed_images):
            # OpenCV детекция
            cv_results = cls.detect_with_opencv(processed_img)
            for result in cv_results:
                result['preprocessing'] = f'processed_{i}'
                all_results.append(result)
            
            # pyzbar детекция
            pyzbar_results = cls.detect_with_pyzbar(processed_img)
            for result in pyzbar_results:
                result['preprocessing'] = f'processed_{i}'
                all_results.append(result)
            
            # 3. Повернутые варианты только для первых обработанных изображений
            if i < 4:  # Ограничиваем количество поворотов
                rotated_variants = cls.rotate_image_variants(processed_img)
                for j, rotated_img in enumerate(rotated_variants[1:], 1):  # Пропускаем оригинал
                    cv_rot_results = cls.detect_with_opencv(rotated_img)
                    for result in cv_rot_results:
                        result['preprocessing'] = f'processed_{i}_rotated_{j}'
                        all_results.append(result)
        
        # Удаляем дубликаты и сортируем по уверенности
        unique_results = []
        seen_data = set()
        
        for result in all_results:
            if result['data'] not in seen_data:
                seen_data.add(result['data'])
                unique_results.append(result)
        
        # Сортируем по уверенности
        unique_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        logger.info(f"Найдено {len(unique_results)} уникальных QR кодов из {len(all_results)} попыток")
        
        return unique_results

processor = AdvancedQRProcessor()

@app.get("/")
async def root():
    """Основная страница API"""
    return {
        "message": "Advanced QR Code Scanner API",
        "version": "2.0.0",
        "features": [
            "Множественная предобработка изображений",
            "Поддержка OpenCV и pyzbar",
            "Автоматическая коррекция поворота",
            "Улучшение контраста и резкости",
            "Адаптивная бинаризация"
        ],
        "endpoints": {
            "/scan": "POST - полное сканирование с детальной информацией",
            "/scan-simple": "POST - простое сканирование (только текст)",
            "/scan-advanced": "POST - продвинутое сканирование с отладочной информацией", 
            "/scan-extreme": "POST - экстремальное сканирование для самых сложных случаев",
            "/scan-screen-photo": "POST - специальное сканирование фотографий экранов",
            "/health": "GET - проверка состояния сервиса"
        }
    }

@app.get("/demo.html")
async def demo_page():
    """Демо страница для тестирования"""
    return FileResponse("demo.html")

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "message": "Advanced QR Scanner service is running",
        "pyzbar_available": PYZBAR_AVAILABLE,
        "opencv_available": True
    }

@app.post("/scan-screen-photo")
async def scan_screen_photo(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Специальное сканирование для фотографий экранов с QR кодами
    Обрабатывает блики, искажения перспективы и низкий контраст
    """
    content_type = getattr(file, 'content_type', None)
    if content_type and not content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Начинаем сканирование фото экрана {image.size}")
        
        # Сохраняем исходное изображение
        debug_dir = "screen_debug"
        os.makedirs(debug_dir, exist_ok=True)
        original_path = f"{debug_dir}/original_screen.jpg"
        image.save(original_path)
        
        # Конвертируем в numpy array
        img_array = np.array(image)
        gray_original = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        results = []
        attempt_count = 0
        
        # Специальные методы для фотографий экранов
        screen_variants = []
        
        # 1. Оригинал
        screen_variants.append((gray_original, "original"))
        
        # 2. Удаление бликов - морфологическое закрытие
        try:
            # Находим яркие области (блики)
            _, bright_mask = cv2.threshold(gray_original, 200, 255, cv2.THRESH_BINARY)
            
            # Расширяем маску бликов
            kernel = np.ones((5,5), np.uint8)
            bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)
            
            # Заполняем блики соседними пикселями
            deglared = cv2.inpaint(gray_original, bright_mask, 3, cv2.INPAINT_TELEA)
            screen_variants.append((deglared, "deglare"))
        except:
            pass
        
        # 3. Коррекция перспективы - автоматический поиск прямоугольника экрана
        try:
            # Размытие для удаления шума
            blurred = cv2.GaussianBlur(gray_original, (5, 5), 0)
            
            # Поиск краев
            edges = cv2.Canny(blurred, 50, 150)
            
            # Поиск контуров
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Ищем наибольший прямоугольный контур
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Найден четырехугольник
                    # Упорядочиваем точки
                    points = approx.reshape(4, 2)
                    
                    # Находим размеры целевого прямоугольника
                    width = max(
                        np.linalg.norm(points[0] - points[1]),
                        np.linalg.norm(points[2] - points[3])
                    )
                    height = max(
                        np.linalg.norm(points[0] - points[3]),
                        np.linalg.norm(points[1] - points[2])
                    )
                    
                    # Целевые точки для прямоугольника
                    dst_points = np.array([
                        [0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]
                    ], dtype=np.float32)
                    
                    # Вычисляем матрицу трансформации
                    matrix = cv2.getPerspectiveTransform(
                        points.astype(np.float32), dst_points
                    )
                    
                    # Применяем коррекцию перспективы
                    corrected = cv2.warpPerspective(
                        gray_original, matrix, (int(width), int(height))
                    )
                    
                    screen_variants.append((corrected, "perspective_corrected"))
                    break
        except:
            pass
        
        # 4. Повышение контраста специально для темных экранов
        try:
            # CLAHE с агрессивными параметрами
            clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray_original)
            screen_variants.append((enhanced, "high_contrast"))
            
            # Гамма коррекция для осветления
            gamma = 0.5  # Осветление
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(gray_original, gamma_table)
            screen_variants.append((gamma_corrected, "gamma_corrected"))
        except:
            pass
        
        # 5. Специальная обработка для QR кодов на экранах
        try:
            # Адаптивная бинаризация с большими блоками
            adaptive = cv2.adaptiveThreshold(
                gray_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 10
            )
            screen_variants.append((adaptive, "adaptive_large_blocks"))
            
            # Инверсия (светлый QR на темном фоне)
            inverted = cv2.bitwise_not(gray_original)
            screen_variants.append((inverted, "inverted"))
            
            # Эквализация гистограммы
            equalized = cv2.equalizeHist(gray_original)
            screen_variants.append((equalized, "equalized"))
        except:
            pass
        
        # 6. Комбинированная обработка
        try:
            # Сначала убираем блики, потом повышаем контраст
            _, bright_mask = cv2.threshold(gray_original, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3), np.uint8)
            bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
            deglared = cv2.inpaint(gray_original, bright_mask, 3, cv2.INPAINT_TELEA)
            
            # Повышаем контраст
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
            final_processed = clahe.apply(deglared)
            screen_variants.append((final_processed, "deglare_plus_contrast"))
        except:
            pass
        
        # 7. Масштабирование для лучшего распознавания
        for scale in [0.5, 1.5, 2.0]:
            try:
                h, w = gray_original.shape
                new_w, new_h = int(w * scale), int(h * scale)
                if 100 < new_w < 3000 and 100 < new_h < 3000:
                    scaled = cv2.resize(gray_original, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    screen_variants.append((scaled, f"scaled_{scale}"))
            except:
                pass
        
        logger.info(f"Создано {len(screen_variants)} специальных вариантов для экрана")
        
        # Тестируем каждый вариант
        for variant, method_name in screen_variants:
            attempt_count += 1
            
            # Сохраняем вариант
            variant_path = f"{debug_dir}/variant_{attempt_count:03d}_{method_name}.jpg"
            try:
                cv2.imwrite(variant_path, variant)
            except:
                pass
            
            # Тестируем OpenCV
            try:
                detector = cv2.QRCodeDetector()
                data, vertices, binary = detector.detectAndDecode(variant)
                
                if data:
                    results.append({
                        'data': data,
                        'type': 'QRCODE',
                        'confidence': 1.0,
                        'method': f'opencv_{method_name}',
                        'variant_file': variant_path
                    })
                    logger.info(f"✅ OpenCV {method_name}: {data[:50]}...")
                
                # Также пробуем с улучшенным детектором
                detector_improved = cv2.QRCodeDetector()
                detector_improved.setEpsX(0.5)
                detector_improved.setEpsY(0.5)
                data_improved, _, _ = detector_improved.detectAndDecode(variant)
                
                if data_improved and data_improved != data:
                    results.append({
                        'data': data_improved,
                        'type': 'QRCODE',
                        'confidence': 0.9,
                        'method': f'opencv_improved_{method_name}',
                        'variant_file': variant_path
                    })
                    logger.info(f"✅ OpenCV improved {method_name}: {data_improved[:50]}...")
                
            except Exception as e:
                logger.debug(f"OpenCV {method_name} failed: {e}")
            
            # Тестируем pyzbar
            if PYZBAR_AVAILABLE:
                try:
                    pil_variant = Image.fromarray(variant)
                    qr_codes = pyzbar.decode(pil_variant)
                    for qr in qr_codes:
                        data = qr.data.decode('utf-8')
                        results.append({
                            'data': data,
                            'type': qr.type,
                            'confidence': 1.0,
                            'method': f'pyzbar_{method_name}',
                            'variant_file': variant_path,
                            'rect': {
                                'left': qr.rect.left,
                                'top': qr.rect.top,
                                'width': qr.rect.width,
                                'height': qr.rect.height
                            }
                        })
                        logger.info(f"✅ pyzbar {method_name}: {data[:50]}...")
                        
                except Exception as e:
                    logger.debug(f"pyzbar {method_name} failed: {e}")
        
        # Удаляем дубликаты
        unique_results = []
        seen_data = set()
        
        for result in results:
            data = result['data']
            if data not in seen_data:
                seen_data.add(data)
                unique_results.append(result)
        
        logger.info(f"Сканирование фото экрана завершено: {attempt_count} попыток, {len(unique_results)} уникальных результатов")
        
        if not unique_results:
            return {
                "success": False,
                "message": f"QR код не найден на фото экрана после {attempt_count} специальных методов обработки",
                "data": None,
                "debug_info": {
                    "total_attempts": attempt_count,
                    "debug_directory": debug_dir,
                    "original_saved": original_path,
                    "message": "Проверьте папку screen_debug/ для визуального анализа",
                    "suggestions": [
                        "Попробуйте снять фото более четко",
                        "Избегайте бликов на экране",
                        "Снимайте прямо, а не под углом",
                        "Убедитесь, что QR код четко виден"
                    ]
                }
            }
        
        return {
            "success": True,
            "message": f"QR код найден на фото экрана!",
            "data": unique_results[0]["data"] if len(unique_results) == 1 else [r["data"] for r in unique_results],
            "details": unique_results,
            "debug_info": {
                "total_attempts": attempt_count,
                "successful_variants": len(unique_results),
                "debug_directory": debug_dir,
                "best_method": unique_results[0].get('method', 'unknown'),
                "best_variant_file": unique_results[0].get('variant_file', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка при сканировании фото экрана: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")

@app.post("/scan-extreme")
async def scan_qr_extreme(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Экстремальное сканирование для самых сложных случаев
    Использует сотни методов обработки изображения
    """
    content_type = getattr(file, 'content_type', None)
    if content_type and not content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Начинаем экстремальное сканирование изображения {image.size}")
        
        # Сохраняем исходное изображение для анализа
        debug_dir = "extreme_debug"
        os.makedirs(debug_dir, exist_ok=True)
        original_path = f"{debug_dir}/original_uploaded.jpg"
        image.save(original_path)
        logger.info(f"Исходное изображение сохранено: {original_path}")
        
        # Используем экстремальный алгоритм
        results = []
        attempt_count = 0
        
        # 1. Простые варианты
        simple_variants = [
            (image, "original"),
        ]
        
        # 2. Масштабирование
        w, h = image.size
        for scale in [0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0]:
            try:
                new_w, new_h = int(w * scale), int(h * scale)
                if 50 < new_w < 5000 and 50 < new_h < 5000:
                    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    simple_variants.append((resized, f"scale_{scale}"))
            except:
                pass
        
        # 3. Обрезка для больших изображений
        if w > 600 or h > 600:
            # Центральная область
            crop_size = min(w, h, 800)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            
            try:
                cropped_center = image.crop((left, top, right, bottom))
                simple_variants.append((cropped_center, "crop_center"))
            except:
                pass
            
            # Деление на 4 части
            for i, (x_start, y_start) in enumerate([(0, 0), (w//2, 0), (0, h//2), (w//2, h//2)]):
                try:
                    x_end = min(x_start + w//2 + 100, w)
                    y_end = min(y_start + h//2 + 100, h)
                    if x_end - x_start > 200 and y_end - y_start > 200:
                        cropped = image.crop((x_start, y_start, x_end, y_end))
                        simple_variants.append((cropped, f"crop_quarter_{i}"))
                except:
                    pass
        
        # 4. PIL улучшения
        for contrast in [0.5, 1.5, 2.0, 3.0]:
            try:
                enhanced = ImageEnhance.Contrast(image).enhance(contrast)
                simple_variants.append((enhanced, f"contrast_{contrast}"))
            except:
                pass
        
        for brightness in [0.5, 1.5, 2.0]:
            try:
                enhanced = ImageEnhance.Brightness(image).enhance(brightness)
                simple_variants.append((enhanced, f"brightness_{brightness}"))
            except:
                pass
        
        for sharpness in [2.0, 3.0, 5.0]:
            try:
                enhanced = ImageEnhance.Sharpness(image).enhance(sharpness)
                simple_variants.append((enhanced, f"sharpness_{sharpness}"))
            except:
                pass
        
        # Специальные фильтры
        try:
            simple_variants.append((image.filter(ImageFilter.EDGE_ENHANCE_MORE), "edge_enhance"))
            simple_variants.append((image.filter(ImageFilter.SHARPEN), "sharpen"))
            simple_variants.append((ImageOps.autocontrast(image), "autocontrast"))
            simple_variants.append((ImageOps.equalize(image), "equalize"))
        except:
            pass
        
        logger.info(f"Создано {len(simple_variants)} вариантов для тестирования")
        
        # Тестируем каждый вариант
        for variant, method_name in simple_variants:
            attempt_count += 1
            
            # Сохраняем вариант
            variant_path = f"{debug_dir}/variant_{attempt_count:03d}_{method_name}.jpg"
            try:
                variant.save(variant_path)
            except:
                pass
            
            # Тестируем OpenCV
            try:
                img_array = np.array(variant)
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                detector = cv2.QRCodeDetector()
                data, vertices, binary = detector.detectAndDecode(gray)
                
                if data:
                    results.append({
                        'data': data,
                        'type': 'QRCODE',
                        'confidence': 1.0,
                        'method': f'opencv_{method_name}',
                        'variant_file': variant_path
                    })
                    logger.info(f"✅ OpenCV {method_name}: {data[:50]}...")
                
            except Exception as e:
                logger.debug(f"OpenCV {method_name} failed: {e}")
            
            # Тестируем pyzbar
            if PYZBAR_AVAILABLE:
                try:
                    qr_codes = pyzbar.decode(variant)
                    for qr in qr_codes:
                        data = qr.data.decode('utf-8')
                        results.append({
                            'data': data,
                            'type': qr.type,
                            'confidence': 1.0,
                            'method': f'pyzbar_{method_name}',
                            'variant_file': variant_path,
                            'rect': {
                                'left': qr.rect.left,
                                'top': qr.rect.top,
                                'width': qr.rect.width,
                                'height': qr.rect.height
                            }
                        })
                        logger.info(f"✅ pyzbar {method_name}: {data[:50]}...")
                        
                except Exception as e:
                    logger.debug(f"pyzbar {method_name} failed: {e}")
        
        # Удаляем дубликаты
        unique_results = []
        seen_data = set()
        
        for result in results:
            data = result['data']
            if data not in seen_data:
                seen_data.add(data)
                unique_results.append(result)
        
        logger.info(f"Экстремальное сканирование завершено: {attempt_count} попыток, {len(unique_results)} уникальных результатов")
        
        if not unique_results:
            return {
                "success": False,
                "message": f"QR код не найден даже после экстремального сканирования ({attempt_count} попыток)",
                "data": None,
                "debug_info": {
                    "total_attempts": attempt_count,
                    "debug_directory": debug_dir,
                    "original_saved": original_path,
                    "message": "Проверьте папку extreme_debug/ для визуального анализа"
                }
            }
        
        return {
            "success": True,
            "message": f"QR код найден с помощью экстремального сканирования!",
            "data": unique_results[0]["data"] if len(unique_results) == 1 else [r["data"] for r in unique_results],
            "details": unique_results,
            "debug_info": {
                "total_attempts": attempt_count,
                "successful_variants": len(unique_results),
                "debug_directory": debug_dir,
                "best_method": unique_results[0].get('method', 'unknown'),
                "best_variant_file": unique_results[0].get('variant_file', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка при экстремальном сканировании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")

@app.post("/scan-advanced")
async def scan_qr_advanced(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Продвинутое сканирование с максимальными усилиями для распознавания
    """
    content_type = getattr(file, 'content_type', None)
    if content_type and not content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Начинаем продвинутое сканирование изображения {image.size}")
        
        # Используем продвинутый процессор
        results = processor.scan_qr_comprehensive(image)
        
        if not results:
            return {
                "success": False,
                "message": "QR код не найден после всех попыток обработки",
                "data": None,
                "attempts_made": "comprehensive_processing"
            }
        
        return {
            "success": True,
            "message": f"Найдено {len(results)} QR код(ов) с помощью продвинутой обработки",
            "data": results[0]["data"] if len(results) == 1 else [r["data"] for r in results],
            "details": results,
            "processing_info": {
                "total_attempts": len(results),
                "best_method": results[0].get('method', 'unknown'),
                "best_preprocessing": results[0].get('preprocessing', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка при продвинутом сканировании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")

@app.post("/scan")
async def scan_qr_code(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Стандартное сканирование QR кода с улучшенной обработкой
    """
    content_type = getattr(file, 'content_type', None)
    if content_type and not content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Используем улучшенный, но более быстрый алгоритм
        results = processor.scan_qr_comprehensive(image)
        
        if not results:
            return {
                "success": False,
                "message": "QR код не найден на изображении",
                "data": None
            }
        
        # Берем только лучший результат для стандартного endpoint
        best_result = results[0]
        
        return {
            "success": True,
            "message": f"QR код успешно распознан",
            "data": best_result["data"],
            "details": [{
                "data": best_result["data"],
                "type": best_result.get("type", "QRCODE"),
                "method": best_result.get("method", "unknown"),
                "confidence": best_result.get("confidence", 1.0)
            }]
        }
        
    except Exception as e:
        logger.error(f"Ошибка при сканировании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")

@app.post("/scan-simple")
async def scan_qr_code_simple(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Упрощенная версия - возвращает только строку из первого найденного QR кода
    """
    content_type = getattr(file, 'content_type', None)
    if content_type and not content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        results = processor.scan_qr_comprehensive(image)
        
        if not results:
            raise HTTPException(status_code=404, detail="QR код не найден на изображении")
        
        return {"result": results[0]["data"]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при простом сканировании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main_advanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
