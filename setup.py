from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rufus-web-extractor',
    version='0.1.0',
    author="Butchi Venkatesh Adari",
    author_email="butchivenkatesh.a@gmail.com",
    description='An AI-powered web data extraction tool for RAG systems',
    keywords="web scraping, ai, llm, rag",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/VenkateshRoshan/rufus-web-extractor",
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'requests>=2.31.0',
        'beautifulsoup4>=4.12.2',
        'openai>=1.3.0',
        'selenium>=4.15.0',
        'webdriver-manager>=4.0.0',
        'python-dotenv>=1.0.0',
        'robotexclusionrulesparser',
        'colorama',
        'langchain',
        'langchain-openai',
        'langchain_community',
        'chromadb',
        'sentence_transformers',
        'python-dotenv',
        'pyyaml',
        'ollama',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'mypy>=1.3.0',
            'black>=23.3.0',
            'isort>=5.12.0',
        ],
        'test': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: General',
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'rufus-web-extractor=rufus.main:main',
            'rufus-api=rufus.api:main',
            'rufus-gradio=rufus.gradio_app:main',
        ],
    },
    include_package_data=True,
    package_data={
        'rufus': ['config/*.yaml', 'templates/*'],
    },
)