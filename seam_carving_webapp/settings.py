"""
Django settings for seam_carving_webapp project.

Generated by 'django-admin startproject' using Django 4.2.6.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import django_heroku
import uuid

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/


# SECURITY WARNING: keep the secret key used in production secret!
load_dotenv()
SECRET_KEY = os.getenv('SEAM_CARVING_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['seam-app-65be144be81b.herokuapp.com']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'seam',
    'celery_progress',
    'django_celery_results',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

ROOT_URLCONF = 'seam_carving_webapp.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
            ],
        },
    },
]

WSGI_APPLICATION = 'seam_carving_webapp.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
    # 'default': {
    #     'ENGINE': os.getenv('MYSQL_ENGINE'),
    #     'NAME': os.getenv('MYSQL_NAME'),
    #     'USER': os.getenv('MYSQL_USER'),
    #     'PASSWORD': os.getenv('MYSQL_PASSWORD'),
    #     'HOST': os.getenv('MYSQL_HOST'),
    #     'PORT': os.getenv('MYSQL_PORT'),
    # }

    #     'default': {
    #     'ENGINE': os.getenv('MYSQL_ENGINE'),
    #     'NAME': os.getenv('MYSQL_NAME'),
    #     'USER': os.getenv('MYSQL_USER'),
    #     'PASSWORD': os.getenv('MYSQL_PASSWORD'),
    #     'HOST': os.getenv('MYSQL_HOST'),
    #     'PORT': os.getenv('MYSQL_PORT'),
    # }
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

# static main directory routing 
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFileStorage'


# celery configuration
# CELERY_BROKER_URL = "redis://localhost/0"
# CELERY_BROKER_URL = os.getenv('REDIS_CLOUD_URL')
CELERY_BROKER_URL = os.getenv('REDIS_HEROKU_URL')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_BACKEND = 'django-db'



# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
django_heroku.settings(locals())

if DATABASES['default'].get('OPTIONS'):
    del DATABASES['default']['OPTIONS']['sslmode']

# LOCAL_STORAGE_VAL = True
LOCAL_STORAGE_VAL = False
STORE_AWS_LOCAL = True

if LOCAL_STORAGE_VAL:
    MEDIA_URL = '/media/'
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

else:
    # DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_SIGNATURE_NAME = os.getenv('AWS_S3_SIGNATURE_NAME')
    AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')




INPUT_PATH  = os.path.join(BASE_DIR, "media/input")
OUTPUT_PATH = os.path.join(BASE_DIR, "media/output")
SEAMS_PATH  = os.path.join(BASE_DIR, "media/seams")
VIDEO_PATH  = os.path.join(BASE_DIR, "media/video")
VIDEO_PATH_AWS  = os.path.join(BASE_DIR, "media")
# VIDEO_PATH2  = os.path.join(BASE_DIR, "media/aws_video")
BUCKET_NAME = 'seam-project'

