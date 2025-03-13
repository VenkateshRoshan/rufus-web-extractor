from setuptools import setup, find_packages

setup(
    name="rufus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "openai>=1.3.0",
        "selenium>=4.15.0",
        "webdriver-manager>=4.0.0",
        "python-dotenv>=1.0.0",
    ],
    author="Butchi Venkatesh Adari",
    author_email="butchivenkatesh.a@gmail.com",
    description="Rufus: Intelligent web data extraction for RAG systems",
    keywords="web scraping, ai, llm, rag",
    url="https://github.com/VenkateshRoshan/rufus-web-extractor",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
)
