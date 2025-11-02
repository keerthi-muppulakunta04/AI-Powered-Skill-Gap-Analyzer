🌟 AI Powered Skill Gap Analyzer 🌟

Author: Keerthi Muppulakunta | Student ID: 17

An AI-driven tool to analyze resumes and job descriptions, extract skills, compare skill gaps, and generate personalized learning paths. Perfect for job seekers, recruiters, and HR professionals who want to optimize talent matching.

🚀 Features

Multi-format document support: PDF, DOCX, TXT (with OCR for scanned PDFs using pytesseract & pdf2image)

Skill extraction: NLP-based extraction using SpaCy, keyword matching, and fuzzy logic

Embeddings & similarity: Calculates skill similarity using Sentence-BERT embeddings

Interactive visualizations with Plotly:

Pie charts for overall skill match

Bubble & radar charts for skill distribution and category coverage

Heatmaps and bar charts for missing and partial skills

Gap analysis: Categorizes skills into Matched, Partial, and Missing

Reports: Export analysis as TXT, CSV, JSON, or PDF

Personalized learning path: Recommends courses/resources based on missing skills

Customizable thresholds: Adjust strong/partial match levels and confidence filters

The app provides an interactive dashboard for uploading resumes & job descriptions, extracting skills, and visualizing gaps.

📂 Supported File Types
Type	Description
PDF	Standard PDFs or scanned PDFs (OCR enabled)
DOCX	Microsoft Word documents
TXT	Plain text files
🔧 Technology Stack

Python 3.10+

Streamlit: Interactive UI

SpaCy: NLP & skill extraction

Sentence-Transformers: Embeddings & similarity

PyPDF2 / docx / pytesseract: File parsing & OCR

Plotly: Interactive visualizations

FuzzyWuzzy: Fuzzy skill matching

Pandas / Numpy: Data processing

ReportLab: PDF report generation

📝 Workflow

Upload documents

Upload resumes and job descriptions side by side

Preview & normalize text

Clean, redact sensitive info, and preview extracted text

Assign roles

Tag which documents are resumes or job descriptions

Skill extraction & analysis

Extract skills using NLP and match JD vs Resume skills

Gap analysis & visualizations

Explore matched, partial, and missing skills

Generate interactive plots (pie, radar, heatmap, bar charts)

Generate reports

Export detailed analysis in TXT, CSV, JSON, or PDF

Personalized learning path

Recommended courses/resources to fill skill gaps

📊 Example Visualizations

Overall Skill Match: Pie chart showing matched, partial, and missing skills

Missing Skills Bar Chart: Highlights top skills to improve

Radar Chart: Category-wise skill coverage

Similarity Heatmap: Shows correspondence between resume and JD skills

⚙️ Configuration

Adjust Sentence-BERT model

Customize Strong / Partial match thresholds

Apply confidence filters to skill extraction

Clear session for fresh analysis

🌟 Highlights

Handles large files (up to 100MB)

OCR fallback for scanned PDFs

Flexible learning path recommendations

Easy to extend skill database with new categories

Interactive and intuitive UI for quick insights

🛠️ Future Enhancements

Add support for LinkedIn profile scraping

Include resume scoring & ranking for multiple candidates

Integration with job portals for automated JD fetching

Advanced AI-powered skill recommendations

📄 References / Dependencies

Streamlit

SpaCy

Sentence-Transformers

PyPDF2

python-docx

pytesseract

FuzzyWuzzy

Plotly

ReportLab

Conclusion

The AI Powered Skill Gap Analyzer provides a comprehensive, automated approach to identify and visualize skill gaps between resumes and job descriptions. By combining NLP, embeddings, fuzzy matching, and interactive visualizations, it enables job seekers, recruiters, and HR professionals to:

Quickly extract and categorize skills from multiple document formats (PDF, DOCX, TXT).

Assess skill coverage and gaps with an intuitive similarity analysis.

Visualize matches, partial matches, and missing skills using charts, radar plots, and heatmaps.

Generate actionable learning paths tailored to missing or underrepresented skills.

Export detailed reports in TXT, CSV, JSON, and PDF formats for further analysis.

This tool empowers users to make data-driven decisions for career development, recruitment, and training, bridging the gap between candidate capabilities and job requirements efficiently.
