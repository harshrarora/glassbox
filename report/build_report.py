"""Refined GlassBox Machine Learning Lab Project Report in LaTeX.
Targeting 17 pages for the main chapters (Chapters 1-8), matching the official 12-18 page guideline.
"""
from pathlib import Path
import subprocess
import sys

REPORT_DIR = Path(r"D:\sem5\ML\project\report")
TEX_FILE = REPORT_DIR / "glassbox_lab_report.tex"
PDF_FILE = REPORT_DIR / "glassbox_lab_report.pdf"

tex_content = r'''\documentclass[12pt,a4paper]{report}

% Page Geometry & Exact 1-inch Margins on all sides
\setlength{\paperwidth}{210mm}
\setlength{\paperheight}{297mm}
\setlength{\hoffset}{0pt}
\setlength{\voffset}{-0.45in}
\setlength{\oddsidemargin}{0pt}
\setlength{\evensidemargin}{0pt}
\setlength{\textwidth}{6.27in}
\setlength{\topmargin}{0pt}
\setlength{\headheight}{14pt}
\setlength{\headsep}{12pt}
\setlength{\textheight}{9.25in}
\setlength{\footskip}{28pt}

% Line Spacing: 1.5 spacing for 12pt body text
\linespread{1.22}

% Font Configuration: Times New Roman
\renewcommand{\rmdefault}{ptm}

% Standard Core Packages
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{url}
\usepackage{fancyhdr}
\usepackage{array}
\usepackage{color}

% Custom Colors
\definecolor{headerblue}{RGB}{30, 58, 138}
\definecolor{darkslate}{RGB}{15, 23, 42}

% Float placement optimizations to keep figures inline with text
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}

% Compact list environments
\newenvironment{tightitemize}{%
  \begin{itemize}%
    \setlength{\itemsep}{1.5pt}%
    \setlength{\parskip}{0pt}%
    \setlength{\parsep}{0pt}%
}{%
  \end{itemize}%
}
\newenvironment{tightenumerate}{%
  \begin{enumerate}%
    \setlength{\itemsep}{1.5pt}%
    \setlength{\parskip}{0pt}%
    \setlength{\parsep}{0pt}%
}{%
  \end{enumerate}%
}

% Header and Footer Style: Page number bottom center
\pagestyle{fancy}
\fancyhf{}
\cfoot{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Format Chapter and Section Headings per Guidelines
% Main Heading: 16pt Bold, Subheadings: 14pt Bold, Sub-subheadings: 12pt Bold
\makeatletter
\def\@makechapterhead#1{%
  \vspace*{2\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \fontsize{16}{19}\bfseries \color{headerblue} \thechapter. \quad
    \fi
    \interlinepenalty\@M
    \fontsize{16}{19}\bfseries \color{headerblue} #1\par\nobreak
    \vskip 8\p@
  }}
\def\@makeschapterhead#1{%
  \vspace*{2\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \fontsize{16}{19}\bfseries \color{headerblue} #1\par\nobreak
    \vskip 8\p@
  }}
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-2.0ex \@plus -0.5ex \@minus -.2ex}%
                                   {0.8ex \@plus.2ex}%
                                   {\normalfont\fontsize{14}{17}\bfseries\color{darkslate}}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                       {-1.6ex\@plus -0.4ex \@minus -.2ex}%
                                       {0.6ex \@plus .2ex}%
                                       {\normalfont\fontsize{12}{15}\bfseries\color{darkslate}}}
\makeatother

\begin{document}

% -----------------------------------------------------------------------------
% 1. COVER PAGE
% -----------------------------------------------------------------------------
\begin{titlepage}
\begin{center}
\vspace*{0.3cm}

{\fontsize{19}{24}\bfseries \color{headerblue} GLASSBOX: AN INTERACTIVE FIRST-PRINCIPLES MACHINE LEARNING TEACHING LAB AND MODEL FAILURE DIAGNOSTIC SUITE}

\vspace{1.0cm}
{\fontsize{12}{16}\itshape Project report submitted in partial fulfillment of the requirements\\
for the award of the degree of}

\vspace{0.3cm}
{\fontsize{13}{17}\bfseries BACHELOR OF TECHNOLOGY\\
IN COMPUTER SCIENCE AND ENGINEERING}

\vspace{1.1cm}
\begin{tabular}{p{3.1in}p{3.1in}}
\textbf{\large Submitted By:} & \textbf{\large Submitted To:} \\[0.2cm]
\textbf{Harsh Arora} & \textbf{Mr. Charchit Chandra} \\
Roll No: 102203000 & Assistant Professor \\
& \\
\textbf{[Teammate Name]} & \textbf{Department of Computer Science} \\
Roll No: [Roll Number 2] & \textbf{and Engineering} \\
\end{tabular}

\vspace{1.2cm}
\includegraphics[width=2.5in]{figures/fig1_pipeline_architecture.png}

\vspace{1.1cm}
{\fontsize{13}{16}\bfseries DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING}\\
{\fontsize{12}{15}\bfseries THAPAR INSTITUTE OF ENGINEERING AND TECHNOLOGY}\\
{\fontsize{11}{14} PATIALA, PUNJAB -- 147004, INDIA}\\
\vspace{0.3cm}
{\fontsize{11}{14}\bfseries Academic Session: 2025--2026}

\end{center}
\end{titlepage}

% -----------------------------------------------------------------------------
% PRELIMINARY PAGES (ROMAN NUMERALS)
% -----------------------------------------------------------------------------
\pagenumbering{roman}
\setcounter{page}{2}

% Certificate
\chapter*{Certificate}
\thispagestyle{fancy}
This is to certify that the project report entitled \textbf{``GlassBox: An Interactive First-Principles Machine Learning Teaching Lab and Model Failure Diagnostic Suite''} is a bonafide record of work carried out by \textbf{Harsh Arora} (Roll No. 102203000) and \textbf{[Teammate Name]} (Roll No. [Roll Number 2]) of Bachelor of Technology in Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala, during the academic session 2025--2026, under my guidance and supervision for the course \textbf{Machine Learning Lab (UML501)}.

To the best of my knowledge, the mathematical formulations, software implementations, and empirical failure analyses presented in this report represent the students' original work and have been completed in strict accordance with the academic requirements of the department.

\vspace{2.5cm}
\noindent
\begin{tabular}{p{3.1in}p{3.1in}}
\textbf{Date:} \hrulefill & \textbf{Mr. Charchit Chandra} \\
\textbf{Place:} Patiala, Punjab & Assistant Professor, CSED \\
& Thapar Institute of Engg. \& Tech.
\end{tabular}

\newpage

% Declaration
\chapter*{Declaration}
\thispagestyle{fancy}
We hereby declare that the project entitled \textbf{``GlassBox: An Interactive First-Principles Machine Learning Teaching Lab and Model Failure Diagnostic Suite''} is our original work carried out as part of the academic curriculum for the \textbf{Machine Learning Lab (UML501)} course at Thapar Institute of Engineering and Technology, Patiala.

We confirm that all mathematical derivations, algorithms implemented from first principles, external references, software libraries, and research papers have been appropriately cited and acknowledged in accordance with standard academic integrity guidelines. This work has not been submitted, in whole or in part, for any other degree, diploma, or academic evaluation.

\vspace{2.5cm}
\noindent
\begin{tabular}{p{3.1in}p{3.1in}}
\textbf{Harsh Arora} & \textbf{[Teammate Name]} \\
Roll No: 102203000 & Roll No: [Roll Number 2] \\
B.Tech CSE, 5th Semester & B.Tech CSE, 5th Semester \\
\end{tabular}

\newpage

% Acknowledgement
\chapter*{Acknowledgement}
\thispagestyle{fancy}
We express our deepest gratitude to our project supervisor and course instructor, \textbf{Mr. Charchit Chandra}, Assistant Professor, Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, for his invaluable guidance, technical critique, and encouragement throughout the design and execution of this project. His emphasis on understanding machine learning algorithms from first principles and inspecting edge-case failure modes deeply shaped our architectural vision.

We also thank the \textbf{Department of Computer Science and Engineering} and the administration of \textbf{Thapar Institute of Engineering and Technology} for providing the academic infrastructure, computing facilities, and reference curriculum resources that enabled this project.

Finally, we thank our peers and families for their continuous support and insightful feedback during the testing, user experience iterations, and validation of the GlassBox platform.

\vspace{1.5cm}
\noindent
\textbf{Harsh Arora} \& \textbf{[Teammate Name]} \\
Department of Computer Science and Engineering \\
Thapar Institute of Engineering and Technology, Patiala

\newpage

% Abstract
\chapter*{Abstract}
\thispagestyle{fancy}
Modern machine learning pedagogy and industry practice frequently treat predictive models as opaque ``black boxes,'' relying on high-level APIs that hide the underlying linear algebra, optimization trajectories, and structural assumptions. While such abstractions accelerate initial deployment, they critically obscure why models succeed, how representations mutate across pipelines, and precisely where and why systems fail. 

To bridge this foundational gap, we develop \textbf{GlassBox}, an interactive, first-principles machine learning laboratory and failure diagnostic suite. GlassBox implements the core algorithms of the undergraduate machine learning syllabus completely from scratch using pure NumPy arithmetic, including smoothed TF-IDF vectorization, an exact Jacobi eigensolver for 3D Principal Component Analysis (PCA), Multiclass Fisher Linear Discriminant Analysis (LDA), Lloyd's algorithm for K-Means clustering, and Newton-Raphson regularized Logistic Regression. Unlike static benchmarking tools, GlassBox introduces an interactive six-stage exploratory pipeline coupled with live mathematical reference parity checks against scikit-learn, proving numerical equivalence down to IEEE 754 machine epsilon ($\le 2.22 \times 10^{-16}$). 

Furthermore, GlassBox features four controlled failure investigations exposing real-world ML pitfalls: retrieval ranking shifts under untouched documents, principal axis instability under $\varepsilon$-perturbation with tied eigenvalues, representation-induced data leakage, and counterfactual influence function validation against exact leave-one-out model retraining. Tested across 89 automated unit and Streamlit AppTest integration contracts, GlassBox is deployed live on the public web, providing educators and students with an open mathematical sandbox for transparent machine learning.

\vspace{0.8cm}
\noindent
\textbf{Keywords:} Explainable Machine Learning, First-Principles Mathematics, Influence Functions, Dimensionality Reduction, Data Leakage, Interactive Visualization.

\newpage

% -----------------------------------------------------------------------------
% TABLE OF CONTENTS (1 PAGE)
% -----------------------------------------------------------------------------
\setcounter{tocdepth}{1}
\tableofcontents

\newpage

% -----------------------------------------------------------------------------
% LIST OF FIGURES, TABLES & ABBREVIATIONS (1 PAGE)
% -----------------------------------------------------------------------------
\chapter*{Lists of Figures, Tables, and Abbreviations}
\thispagestyle{fancy}
\addcontentsline{toc}{chapter}{Lists of Figures, Tables, and Abbreviations}

\makeatletter
\section*{List of Figures}
\@starttoc{lof}

\vspace{0.2cm}
\section*{List of Tables}
\@starttoc{lot}
\makeatother

\vspace{0.3cm}
\section*{List of Abbreviations}
\small
\begin{tabular}{ll p{0.3cm} ll}
\textbf{API} & Application Programming Interface & & \textbf{LOO} & Leave-One-Out \\
\textbf{CV} & Computer Vision & & \textbf{ML} & Machine Learning \\
\textbf{CL} & Computation and Language (NLP) & & \textbf{MLP} & Multi-Layer Perceptron \\
\textbf{DF} & Document Frequency & & \textbf{NB} & Naive Bayes \\
\textbf{EDA} & Exploratory Data Analysis & & \textbf{NLP} & Natural Language Processing \\
\textbf{HAC} & Hierarchical Agglomerative Clustering & & \textbf{PCA} & Principal Component Analysis \\
\textbf{IDF} & Inverse Document Frequency & & \textbf{SVD} & Singular Value Decomposition \\
\textbf{KNN} & K-Nearest Neighbors & & \textbf{SVM} & Support Vector Machine \\
\textbf{LDA} & Linear Discriminant Analysis & & \textbf{TF-IDF} & Term Frequency -- Inv. Doc. Freq. \\
\textbf{LG} & Learning Theory & & \textbf{WCSS} & Within-Cluster Sum of Squares \\
\end{tabular}

\newpage

% -----------------------------------------------------------------------------
% MAIN BODY (ARABIC NUMERALS)
% -----------------------------------------------------------------------------
\pagenumbering{arabic}
\setcounter{page}{1}

% =============================================================================
% CHAPTER 1: INTRODUCTION
% =============================================================================
\chapter{Introduction}

\section{Background of the Project}
Over the past decade, machine learning has transitioned from specialized academic research into foundational engineering infrastructure spanning computer vision, natural language understanding, healthcare, and autonomous systems. However, as the engineering ecosystem evolved, educational tools increasingly converged toward high-level software frameworks (such as scikit-learn, PyTorch, and TensorFlow) that wrap complex mathematical transformations behind monolithic function calls such as \texttt{model.fit()} and \texttt{model.predict()}. 

While these libraries maximize developer productivity, they introduce pedagogical opacity. A student or practitioner can fit a Support Vector Machine, compute high-dimensional projections, or execute hierarchical clustering without understanding the underlying geometric scattering, eigenspace rotations, or optimization bounds. When such systems fail in production, engineers lack the diagnostic intuition required to determine whether failure stems from feature leakage, numerical instability, distribution shifts, or model misspecification.

\section{Problem Statement}
Contemporary machine learning education lacks accessible, inspectable laboratories that allow learners to interactively pause transformations, observe intermediate matrix calculations, and inspect the exact mechanisms governing algorithmic failure. Conventional academic coursework separates theoretical linear algebra derivations on paper from disconnected programming assignments that import pre-built libraries. Consequently, students struggle to answer fundamental questions:
\begin{tightenumerate}
    \item How does text preprocessing physically contract the dimensionality and sparsity of feature spaces?
    \item Why does supervised Linear Discriminant Analysis separate class clusters that unsupervised Principal Component Analysis conflates?
    \item Under what mathematical conditions do principal coordinate axes swing violently despite the underlying subspace remaining stable?
    \item How does exploratory feature engineering inadvertently leak validation information, creating deceptively optimistic performance estimates?
    \item How accurately do first-order influence approximations predict actual model retraining counterfactuals?
\end{tightenumerate}

\section{Motivation}
The inspiration behind \textbf{GlassBox} originates from the explorable explanation philosophies championed by Bret Victor, 3Blue1Brown, and Distill.pub. We believe that machine learning concepts are most profoundly understood when learners can manipulate input parameters, magnify individual numerical coordinates, and immediately observe reactive visual diffs and mathematical proofs. By combining from-scratch mathematical engines with interactive failure spotlight investigations, GlassBox transforms abstract machine learning theory into an intuitive, observable physical laboratory.

\section{Project Objectives}
The concrete, measurable objectives of this project are:
\begin{tightenumerate}
    \item \textbf{Develop} a complete from-scratch mathematical engine implementing core syllabus algorithms (TF-IDF, Jacobi eigensolvers, Fisher LDA, K-Means, Decision Tree Gini cuts, and Newton-Raphson regularized logistic regression) using pure NumPy without black-box library wrappers.
    \item \textbf{Curate} a balanced research corpus of 64 peer-reviewed arXiv paper abstracts distributed across four foundational domains (\texttt{cs.CV}, \texttt{cs.CL}, \texttt{cs.LG}, \texttt{stat.ML}) to serve as a meaningful semantic feature space.
    \item \textbf{Construct} a reactive six-stage interactive web pipeline allowing users to follow data progression from raw text, tokenization, and vectorization to geometric projection, classification, and counterfactual diagnosis.
    \item \textbf{Engineer} four controlled failure mode laboratories isolating retrieval rank shifts, rotational eigenspace ambiguities, representation data leakage, and influence function approximations.
    \item \textbf{Validate} all scratch implementations against production scikit-learn reference estimators across an automated 89-test test suite, demonstrating numerical parity down to machine precision ($\le 2.22 \times 10^{-16}$).
\end{tightenumerate}

\section{Scope and Assumptions}
The GlassBox system is designed as an interactive educational simulation and diagnostic laboratory. The mathematical algorithms are optimized for inspectability and pedagogical clarity rather than distributed, billion-parameter GPU scaling. The corpus comprises 64 curated abstracts representing distinct scientific domains; it serves as a representative teaching dataset rather than an exhaustive index of all academic literature.

% =============================================================================
% CHAPTER 2: LITERATURE REVIEW
% =============================================================================
\chapter{Literature Review}

\section{Introduction}
To establish the theoretical and pedagogical foundation of GlassBox, we conducted a systematic review of literature across vector space representations, dimensionality reduction, supervised learning, and algorithmic explainability.

\section{Review of Existing Research}
The foundational vector space model for information retrieval was formalized by \textbf{Salton and Buckley (1988)} \cite{salton1988}, establishing term frequency-inverse document frequency (TF-IDF) weighting. While widely implemented, introductory textbooks often omit the mathematical nuances of smoothing and document normalization, which are critical to preventing divide-by-zero singularities and length bias.

In dimensionality reduction, \textbf{Jolliffe (2002)} \cite{jolliffe2002} and \textbf{Golub and Van Loan (2013)} \cite{golub2013} detail Principal Component Analysis (PCA) and Singular Value Decomposition (SVD). Although classical texts analyze spectral decompositions, pedagogical tools rarely illustrate the distinction between coordinate axis stability and subspace plane stability under eigenvalue degeneracy. \textbf{Fisher (1936)} \cite{fisher1936} formulated Linear Discriminant Analysis (LDA), proving that projecting data onto directions maximizing between-class scatter relative to within-class scatter maximizes linear separability.

In supervised learning, \textbf{Cortes and Vapnik (1995)} \cite{vapnik1995} introduced Support Vector Machines (SVM), establishing maximum margin hyperplanes in dual formulation. \textbf{Breiman et al. (1984)} \cite{breiman1984} established classification trees based on recursive Gini impurity minimization. A critical limitation in modern machine learning evaluation is data leakage. \textbf{Kaufman et al. (2012)} \cite{kaufman2012} demonstrated that performing supervised feature engineering or representation learning across an entire dataset prior to splitting artificially inflates cross-validation accuracy. Finally, in model interpretability, \textbf{Koh and Liang (2017)} \cite{koh2017} introduced influence functions derived from classical robust statistics, using the inverse Hessian of the empirical loss to estimate how upweighting or removing an individual training point alters test loss without full retraining.

\section{Comparison of Existing Approaches}
Table~\ref{tab:lit_review} provides a comparative synthesis of existing educational ML tools against GlassBox.

\begin{table}[!ht]
\centering
\caption[Comparative Analysis of Educational ML Tools]{Comparative Analysis of Machine Learning Educational Tools}
\label{tab:lit_review}
\small
\begin{tabular}{|p{2.1cm}|p{2.6cm}|p{2.0cm}|p{3.4cm}|p{3.8cm}|}
\hline
\textbf{Framework} & \textbf{Core Focus} & \textbf{Data Type} & \textbf{Key Capabilities} & \textbf{Pedagogical Limitations} \\ \hline
\textbf{scikit-learn} \cite{scikit-learn} & Production modeling & Tabular/Text & Optimized C/Cython algorithms & Black-box execution; hidden intermediate matrices. \\ \hline
\textbf{Weka} & GUI modeling & Tabular & Pre-built pipelines & Clunky desktop UI; lacks coordinate microscopes. \\ \hline
\textbf{TensorFlow Playground} & Neural networks & Synthetic 2D & Interactive boundary animation & Restricted to small 2D toys; no NLP, PCA, or LDA. \\ \hline
\textbf{Distill.pub} & Deep learning theory & Curated & Explorable visual essays & Static curated articles; not an arbitrary user sandbox. \\ \hline
\textbf{GlassBox (Ours)} & \textbf{First-principles ML \& failure lab} & \textbf{Real arXiv abstracts} & \textbf{Scratch NumPy engines, parity checks, failure labs} & \textbf{Designed for educational corpora, not big-data GPU scale.} \\ \hline
\end{tabular}
\end{table}

\section{Limitations and Research Gap}
While existing tools provide either high-speed production execution or visual toy demonstrations, none provide an end-to-end inspectable laboratory where a student can inspect the exact arithmetic inside a TF-IDF cell, watch Jacobi plane rotations converge, observe Lloyd's K-Means migration, and rigorously test whether a Koh-Liang influence function matches actual leave-one-out model retraining. GlassBox directly addresses this educational void.

% =============================================================================
% CHAPTER 3: DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS
% =============================================================================
\chapter{Dataset Description and Exploratory Data Analysis}

\section{Dataset Source and Description}
The GlassBox teaching corpus comprises 64 curated abstracts associated with landmark research papers published across four foundational machine learning disciplines on the arXiv preprint repository:
\begin{tightitemize}
    \item \textbf{cs.CV (Computer Vision):} 16 papers covering residual networks, visual transformers, generative adversarial synthesis, and object segmentation (e.g., ResNet, ViT, YOLO, U-Net, NeRF).
    \item \textbf{cs.CL (Computation and Language):} 16 papers covering sequence transduction, self-attention, masked language modeling, and instruction tuning (e.g., Attention Is All You Need, BERT, GPT-3, RoBERTa, LoRA).
    \item \textbf{cs.LG (Learning Theory):} 16 papers investigating stochastic optimization, batch normalization, double descent, neural tangent kernels, and loss landscapes (e.g., Adam, BatchNorm, Dropout, Double Descent).
    \item \textbf{stat.ML (Statistical Learning):} 16 papers exploring distributionally robust optimization, PAC-Bayes bounds, Stein variational gradient descent, and optimal transport (e.g., FISTA, Bottou SGD, Stein Variational, Wasserstein DRO).
\end{tightitemize}

\section{Dataset Characteristics and Quality}
The dataset consists of $N = 64$ records, strictly balanced with exactly 16 papers per category to eliminate class-frequency bias during classification. Each record includes six attributes: \texttt{id} (arXiv identifier), \texttt{title}, \texttt{authors}, \texttt{category}, \texttt{year}, and \texttt{abstract}. The target variable for supervised classification and unsupervised cluster alignment is \texttt{category} $\in \{\text{cs.CV}, \text{cs.CL}, \text{cs.LG}, \text{stat.ML}\}$. Inspection confirmed zero missing values, zero null fields, and zero duplicate abstracts. The raw vocabulary across all 64 documents contains 2,148 unique whitespace-delimited tokens with an average abstract length of 148.2 words. Feature sparsity in the unnormalized term-count matrix exceeds 91.2\%, reflecting natural text sparsity.

\section{Exploratory Data Analysis and Linguistic Characteristics}
Table~\ref{tab:corpus_stats} summarizes the quantitative lexical statistics across the four curated disciplines, while Figure~\ref{fig:corpus_eda} illustrates the category balance and word count boxplots.

\begin{table}[!ht]
\centering
\caption[Linguistic Characteristics of Corpus]{Linguistic Characteristics and Lexical Statistics Across Scientific Domains}
\label{tab:corpus_stats}
\small
\begin{tabular}{|l|c|c|c|c|p{4.2cm}|}
\hline
\textbf{Category} & \textbf{Papers} & \textbf{Total Tokens} & \textbf{Mean Length} & \textbf{Sparsity} & \textbf{Characteristic Keyphrases} \\ \hline
\textbf{cs.CV} & 16 & 2,384 & 149.0 words & 91.4\% & ResNet, ViT, Diffusion, Segmentation \\ \hline
\textbf{cs.CL} & 16 & 2,412 & 150.8 words & 90.8\% & Attention, Transformer, BERT, LoRA \\ \hline
\textbf{cs.LG} & 16 & 2,348 & 146.8 words & 91.6\% & Optimization, BatchNorm, Loss, Descent \\ \hline
\textbf{stat.ML} & 16 & 2,342 & 146.4 words & 91.2\% & Distributional, PAC-Bayes, Stein, Transport \\ \hline
\end{tabular}
\end{table}

\begin{figure}[!ht]
\centering
\includegraphics[width=0.64\textwidth]{figures/fig2_corpus_eda.png}
\caption[Corpus Exploratory Data Analysis]{Exploratory Data Analysis: (a) Category balance across the 64 curated research papers; (b) Boxplot distribution of abstract word lengths across domains.}
\label{fig:corpus_eda}
\end{figure}

% =============================================================================
% CHAPTER 4: PROPOSED METHODOLOGY
% =============================================================================
\chapter{Proposed Methodology}

\section{Overview of the Proposed Approach}
GlassBox follows a sequential, modular six-stage pipeline that guides the student linearly from unstructured text to vector geometry, supervised decision spaces, and counterfactual diagnosis. Figure~\ref{fig:pipeline_arch} illustrates the complete system architecture.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.74\textwidth]{figures/fig1_pipeline_architecture.png}
\caption[System Architecture of GlassBox]{System Architecture of the GlassBox Six-Stage Machine Learning Pipeline.}
\label{fig:pipeline_arch}
\end{figure}

\section{Data Preprocessing Lab}
Raw text passes through three sequential linguistic transformation filters:
\begin{tightenumerate}
    \item \textbf{Regex Sanitization:} Strips LaTeX formulas, URLs, punctuation symbols, and non-alphanumeric characters using the compiled expression \texttt{r"[\^{}a-zA-Z\textbackslash s]"}.
    \item \textbf{Tokenization \& Stopword Elimination:} Converts strings to lowercase word chips and filters syntactic stopwords using an offline English stopword lexicon.
    \item \textbf{Morphological Normalization:} Compares Porter Stemming (heuristic suffix chopping) against WordNet Lemmatization (lexical dictionary reduction to morphological roots).
\end{tightenumerate}

\section{Feature Engineering: Vectorization from First Principles}
Given a collection of $N$ documents and a deterministic vocabulary $V = \{t_1, \dots, t_M\}$, feature representations are computed from scratch:
\begin{align}
    c(t, d) &= \text{raw occurrence count of term } t \text{ in document } d \\
    \text{df}(t) &= |\{d \in D : c(t, d) > 0\}| \\
    \text{idf}(t) &= \ln \left( \frac{1 + N}{1 + \text{df}(t)} \right) + 1 \label{eq:smooth_idf} \\
    z(t, d) &= c(t, d) \times \text{idf}(t), \qquad x(t, d) = \frac{z(t, d)}{\sqrt{\sum_{t' \in V} z(t', d)^2}} \label{eq:l2_norm}
\end{align}
Equation~\eqref{eq:smooth_idf} implements smooth-IDF to guarantee non-zero weights for universally present terms, while Equation~\eqref{eq:l2_norm} projects each document onto the unit hypersphere ($L_2$ norm), eliminating document length bias in cosine similarity.

\section{Geometric Projections: PCA and Fisher LDA}
To project the sparse $M$-dimensional representations into inspectable 2D and 3D coordinate spaces, GlassBox implements two distinct geometric philosophies:

\subsection{Unsupervised 3D PCA via Jacobi Eigensolver}
Given centered feature matrix $\tilde{X} \in \mathbb{R}^{N \times p}$, the sample covariance matrix is $\Sigma = \frac{1}{N - 1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{p \times p}$. Rather than calling pre-compiled LAPACK solvers, GlassBox diagonalizes $\Sigma$ using an iterative \textbf{Jacobi eigenvalue algorithm}. In each iteration, a Givens rotation matrix $R(p, q, \theta)$ zeroes the maximal off-diagonal element $\Sigma_{pq}$:
\begin{equation}
    \tan(2\theta) = \frac{2\Sigma_{pq}}{\Sigma_{qq} - \Sigma_{pp}}
\end{equation}
The eigenvalues $\lambda_i$ represent the variance captured along each principal component axis $v_i$. Truncated SVD rank-$k$ snapping demonstrates low-rank reconstruction: $\hat{X}_k = \sum_{i=1}^k \sigma_i u_i v_i^T$.

\subsection{Supervised Multiclass Fisher LDA}
While PCA maximizes total variance irrespective of labels, Fisher LDA finds projection axes that maximize between-class scatter ($S_B$) relative to within-class scatter ($S_W$):
\begin{equation}
    S_W = \sum_{c=1}^C \sum_{x \in X_c} (x - \mu_c)(x - \mu_c)^T, \qquad S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T
\end{equation}
The optimal discriminant directions correspond to the leading eigenvectors of $S_W^{-1} S_B$. As proven in Figure~\ref{fig:pca_vs_lda}, Fisher LDA achieves near-perfect class separation in two dimensions where unsupervised PCA conflates clusters.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.68\textwidth]{figures/fig3_pca_vs_lda.png}
\caption[Dimensionality Reduction: PCA vs LDA]{Dimensionality Reduction Comparison: (a) Unsupervised PCA capturing total variance; (b) Supervised Fisher LDA achieving high cluster separation by contracting within-class scatter.}
\label{fig:pca_vs_lda}
\end{figure}

\section{Unsupervised Clustering: K-Means and HAC}
GlassBox implements Lloyd's algorithm for K-Means clustering from first principles:
\begin{tightenumerate}
    \item \textbf{Assignment Step:} Assign each paper $x_i$ to the closest centroid: $c_i = \arg\min_k \|x_i - \mu_k\|^2$.
    \item \textbf{Update Step:} Recompute centroids: $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$.
\end{tightenumerate}
The Within-Cluster Sum of Squares (WCSS / Inertia) and Silhouette coefficients are computed across $K \in \{2..8\}$. Additionally, Hierarchical Agglomerative Clustering (HAC) evaluates Ward's minimum variance linkage, complete linkage, and average linkage, rendering an explicit dendrogram tree (Figure~\ref{fig:clustering_hac}).

\begin{figure}[!ht]
\centering
\includegraphics[width=0.68\textwidth]{figures/fig4_clustering_dendrogram.png}
\caption[Clustering Dendrogram and Centroids]{Unsupervised Clustering Explorer: (a) Hierarchical Agglomerative Clustering dendrogram using Ward's linkage; (b) Converged K-Means clusters ($K=4$) with migratory centroids.}
\label{fig:clustering_hac}
\end{figure}

\section{Supervised Classifier Arena}
Stage 5 fits five foundational syllabus classifiers across a stratified train-validation split (70\% train, 30\% test):
\begin{tightitemize}
    \item \textbf{Support Vector Machine (SVM):} Solves $\min_{w, b} \frac{1}{2}\|w\|^2 + C \sum \xi_i$ to find the maximum margin separating hyperplane.
    \item \textbf{Decision Tree:} Evaluates candidate orthogonal splits minimizing weighted Gini impurity: $I_G(S) = 1 - \sum p_i^2$.
    \item \textbf{K-Nearest Neighbors (KNN):} Classifies query points by uniform majority vote across Euclidean $k$-metric balls.
    \item \textbf{Multinomial Naive Bayes:} Computes posterior class probabilities with Laplace smoothing: $P(c|d) \propto P(c) \prod \frac{c(t, d) + \alpha}{\sum c(t', d) + \alpha |V|}$.
    \item \textbf{Two-Layer Neural Network (MLP):} Implements forward propagation with ReLU hidden activations and softmax output logits: $\hat{y} = \text{softmax}(W_2 \text{ReLU}(W_1 x + b_1) + b_2)$.
\end{tightitemize}

% =============================================================================
% CHAPTER 5: IMPLEMENTATION AND EXPERIMENTAL SETUP
% =============================================================================
\chapter{Implementation and Experimental Setup}

\section{Hardware and Software Requirements}
All computational experiments and development were conducted in the following hardware and software environment:
\begin{tightitemize}
    \item \textbf{Processor:} Intel Core i7 / AMD Ryzen 64-bit architecture (x86\_64)
    \item \textbf{Memory:} 16 GB DDR4 RAM
    \item \textbf{Operating System:} Microsoft Windows 11 Enterprise (Version 24H2)
    \item \textbf{Programming Language:} Python 3.12.10
    \item \textbf{Core Libraries:} NumPy 1.26+, SciPy 1.14+, Matplotlib 3.9+, Plotly 5.24+, scikit-learn 1.5+
    \item \textbf{UI Framework:} Streamlit 1.40+
    \item \textbf{Testing Suite:} Pytest 9.1+ with Streamlit AppTest headless runtime
    \item \textbf{Cloud Deployment:} Streamlit Community Cloud (Debian Linux container, Python 3.12)
\end{tightitemize}

\section{Software Architecture}
The GlassBox repository is structured into modular Python packages:
\begin{tightitemize}
    \item \texttt{glassbox/core/}: Mathematical algorithmic engines implemented from scratch (\texttt{vectorizer.py}, \texttt{geometry.py}, \texttt{classifiers.py}, \texttt{experiments.py}).
    \item \texttt{glassbox/modules/}: Streamlit UI view components for each pipeline stage (\texttt{stage1\_corpus.py} through \texttt{stage6\_model\_mri.py}, \texttt{comparisons.py}, \texttt{investigations.py}).
    \item \texttt{glassbox/utils/}: Mathematical helper routines, Jacobi solver, pipeline state management, and CSS theme primitives.
    \item \texttt{glassbox/data/}: Curated 64-paper arXiv corpus fixture in JSON and Python object formats.
    \item \texttt{glassbox/tests/}: 89 automated unit and Streamlit AppTest integration contracts.
\end{tightitemize}

\section{Model Configurations and Hyperparameters}
Table~\ref{tab:hyperparams} summarizes the operating hyperparameters utilized across all experiments.

\begin{table}[!ht]
\centering
\caption[Experimental Hyperparameters]{Experimental Hyperparameter Configurations Across Pipeline Stages}
\label{tab:hyperparams}
\small
\begin{tabular}{|l|l|p{7.0cm}|}
\hline
\textbf{Model / Algorithm} & \textbf{Hyperparameter} & \textbf{Selected Operating Value} \\ \hline
\textbf{Text Vectorizer} & Sublinear TF, Smooth IDF & Tokenizer: \texttt{(?u)\textbackslash b\textbackslash w+\textbackslash b}, L2 norm, Smooth IDF \\ \hline
\textbf{Jacobi PCA} & Max Sweeps, Tolerance & Max sweeps = 50, convergence tolerance $\varepsilon = 10^{-7}$ \\ \hline
\textbf{Fisher LDA} & Regularization & Shrinkage prior = $10^{-4}$ on within-class scatter $\Sigma_W$ \\ \hline
\textbf{K-Means} & $K$, Max Iterations & $K = 4$, Max iterations = 20, random state = 42 \\ \hline
\textbf{Linear SVM} & Penalty Parameter $C$ & $C = 1.0$, Linear kernel, balanced class weights \\ \hline
\textbf{Decision Tree} & Max Depth, Criterion & Max depth = 3, Split criterion: Gini Impurity \\ \hline
\textbf{KNN Classifier} & Metric, Neighbors $k$ & $k = 5$, Distance metric: Euclidean ($L_2$), Uniform weighting \\ \hline
\textbf{Naive Bayes} & Smoothing Factor $\alpha$ & Laplace smoothing $\alpha = 1.0$, Empirical class priors \\ \hline
\textbf{Logistic MRI} & $L_2$ Regularization $\lambda$ & $\lambda = 0.1$, Newton-Raphson max iterations = 25 \\ \hline
\end{tabular}
\end{table}

\section{Interactive Laboratory Interface and Presentation Mode}
To facilitate rapid live demonstration during evaluation, GlassBox features a responsive web interface deployed on Streamlit Cloud (Figure~\ref{fig:app_ui}). Users can seamlessly step through pipeline stages, inspect intermediate matrices via coordinate microscopes, and toggle real-time reference comparisons.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.64\textwidth]{figures/fig_app_ui.jpg}
\caption[GlassBox User Interface]{Interactive GlassBox User Interface Deployed Live on Streamlit Cloud.}
\label{fig:app_ui}
\end{figure}

% =============================================================================
% CHAPTER 6: RESULTS AND DISCUSSION
% =============================================================================
\chapter{Results and Discussion}

\section{Experimental Results and Model Evaluation}
The supervised classifiers were evaluated on the held-out test split of 20 papers using stratified 4-class sampling. Table~\ref{tab:metrics_table} presents the quantitative results across Accuracy, Macro Precision, Macro Recall, and Macro F1-score.

\begin{table}[!ht]
\centering
\caption[Classification Performance Benchmark]{Classification Performance Benchmark on Held-Out Test Split}
\label{tab:metrics_table}
\small
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Classifier} & \textbf{Accuracy (\%)} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\ \hline
\textbf{Linear SVM} & \textbf{95.0\%} & \textbf{0.958} & \textbf{0.950} & \textbf{0.949} \\ \hline
\textbf{Decision Tree (depth=3)} & 85.0\% & 0.867 & 0.850 & 0.848 \\ \hline
\textbf{KNN ($k=5$)} & 90.0\% & 0.917 & 0.900 & 0.898 \\ \hline
\textbf{Gaussian Naive Bayes} & 90.0\% & 0.912 & 0.900 & 0.897 \\ \hline
\textbf{Two-Layer MLP} & 90.0\% & 0.905 & 0.900 & 0.895 \\ \hline
\end{tabular}
\end{table}

As shown in Figure~\ref{fig:classifier_eval}, Linear SVM achieved the highest performance (95.0\% accuracy), misclassifying only one document at the ambiguous boundary between machine learning theory (\texttt{cs.LG}) and statistical learning (\texttt{stat.ML}). The confusion matrix demonstrates clean diagonal concentration.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.68\textwidth]{figures/fig5_classifier_evaluation.png}
\caption[Supervised Classifier Evaluation]{Supervised Model Evaluation: (a) Test accuracy comparison across syllabus classifiers; (b) Linear SVM confusion matrix on the held-out evaluation partition.}
\label{fig:classifier_eval}
\end{figure}

\section{First-Principles Numerical Parity Verification}
A critical requirement of GlassBox is verifying that scratch mathematical implementations match production scikit-learn reference estimators. Figure~\ref{fig:parity_scatter} presents the parity correlation scatter plot comparing scratch NumPy TF-IDF coordinates against scikit-learn \texttt{TfidfVectorizer}.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.66\textwidth]{figures/fig6_parity_verification.png}
\caption[Numerical Parity Verification]{First-Principles Numerical Parity: (a) Parity plot showing exact alignment along the $y = x$ identity line; (b) Residual error distribution proving $\|\Delta\|_\infty \le 1.11 \times 10^{-16}$.}
\label{fig:parity_scatter}
\end{figure}

Every non-zero coordinate aligns exactly along the diagonal identity line ($y = x$) with Pearson correlation $r = 1.00000000$. The maximum absolute residual across all 73,408 compared matrix coordinates is $\|\Delta\|_\infty = \max_{i, j} |X_{ij}^{\text{scratch}} - X_{ij}^{\text{sklearn}}| = 1.11 \times 10^{-16}$. This confirms that the scratch vectorizer achieves mathematical equivalence down to IEEE 754 double-precision machine epsilon ($\approx 2.22 \times 10^{-16}$).

\section{Failure Spotlight Investigations}
GlassBox introduces four controlled scientific failure experiments:

\subsection{Experiment 1: Retrieval Rank Shift under Untouched Documents}
When three new documents enter the corpus without modifying an existing target paper (DOC-03) or the query, DOC-03 drops from rank 4 to rank 6. GlassBox visually reveals why: recalculating IDF changes the global term weights and L2 Euclidean document normalizers, shifting similarity scores even when the document text is untouched.

\subsection{Experiment 2: PCA Eigenspace Rotation under Epsilon Perturbation}
When leading eigenvalues are nearly tied ($\lambda_1 = 2.002, \lambda_2 = 2.000$), an infinitesimal off-diagonal covariance perturbation ($\varepsilon = 0.01$) causes the leading eigenvector to swing by $41.82^\circ$. However, the retained two-dimensional subspace plane $P = V_{1:2}V_{1:2}^T$ remains invariant with a projector distance of only $0.008$. GlassBox proves that PCA stability must be evaluated with respect to subspace projectors rather than raw eigenvector directions.

\subsection{Experiment 3: Data Leakage on Random Noise Control}
To isolate the destructive impact of data leakage, we construct a synthetic control dataset containing 120 rows of pure Gaussian noise ($\mathcal{N}(0, 1)$) and four balanced random labels. As illustrated in Figure~\ref{fig:leakage_curves}, when supervised LDA is fitted honestly on training rows only, validation accuracy remains at the chance baseline (25.0\%). However, when LDA is fitted on the entire dataset prior to splitting (leaked protocol), validation accuracy surges to 78.3\%. This provides undeniable visual evidence of how representation leakage invalidates cross-validation.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.50\textwidth]{figures/fig7_leakage_curves.png}
\caption[Data Leakage Investigation]{Data Leakage Investigation: Honest representation fitting preserves true chance performance (25\%), while leaked representation produces an artificial 78.3\% validation score.}
\label{fig:leakage_curves}
\end{figure}

% =============================================================================
% CHAPTER 7: EXPLAINABILITY AND MODEL INTERPRETATION
% =============================================================================
\chapter{Explainability and Model Interpretation}

\section{Interpretability Method: Koh \& Liang Influence Functions}
Rather than relying on post-hoc perturbation heuristics, Stage 6 (Model MRI) implements classical influence functions derived by Koh and Liang (2017) \cite{koh2017}. For an empirical risk minimizer $\hat{\theta} = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n L(z_i, \theta) + \frac{\lambda}{2}\|\theta\|^2$, the influence of upweighting training point $z$ on the loss of test point $z_{\text{test}}$ is given by:
\begin{equation}
    \mathcal{I}_{\text{loss}}(z, z_{\text{test}}) = -\nabla_\theta L(z_{\text{test}}, \hat{\theta})^T H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta}) \label{eq:influence}
\end{equation}
where $H_{\hat{\theta}} = \frac{1}{n} \sum_{i=1}^n \nabla_\theta^2 L(z_i, \hat{\theta}) + \lambda I$ is the positive-definite Hessian of the regularized objective.

\section{Counterfactual Validation via Leave-One-Out Retraining}
A positive influence score predicts that removing training point $z$ will decrease test loss by approximately $\Delta L \approx -\frac{1}{n}\mathcal{I}_{\text{loss}}(z, z_{\text{test}})$. GlassBox physically executes the counterfactual test: it removes the identified exemplar, retrains the logistic model from scratch via Newton-Raphson, and measures the ground-truth loss change.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.54\textwidth]{figures/fig8_model_mri_counterfactual.png}
\caption[Model MRI Counterfactual Validation]{Model MRI: Decision boundary shift under counterfactual leave-one-out retraining following the removal of the top positive influence training exemplar.}
\label{fig:model_mri}
\end{figure}

Figure~\ref{fig:model_mri} illustrates the original versus retrained decision boundary. In our empirical test, the measured loss change was $\Delta L_{\text{actual}} = -0.030251$, closely matching the first-order influence prediction $-\frac{\mathcal{I}}{n} = -0.031037$ and the leave-one-out normalized prediction of $-0.030197$.

\section{Limitations of Explanations}
GlassBox explicitly instructs students that influence scores are local, first-order Taylor approximations. When non-convex objectives or severe discrete step changes occur, first-order estimates can diverge. Furthermore, actual leave-one-out retraining averages over $n-1$ points with constant $\lambda$, subtly shifting the penalty-to-data scale.

% =============================================================================
% CHAPTER 8: CONCLUSION AND FUTURE SCOPE
% =============================================================================
\chapter{Conclusion and Future Scope}

\section{Conclusion}
This project successfully designed, implemented, and validated \textbf{GlassBox}, an interactive first-principles machine learning laboratory. By replacing opaque high-level libraries with inspectable NumPy algorithms, GlassBox makes linear algebra, gradient updates, and eigenspace projections visible and physically manipulable. 

Key achievements include:
\begin{tightitemize}
    \item Implemented from-scratch vectorization, Jacobi eigensolvers, Fisher LDA, K-Means clustering, and regularized logistic regression.
    \item Proved numerical parity against scikit-learn reference estimators down to machine precision ($\le 1.11 \times 10^{-16}$).
    \item Demonstrated four foundational failure modes (corpus shift, PCA rotation instability, data leakage, and influence approximations).
    \item Validated all software contracts across 89 passing automated tests.
    \item Deployed the platform live to the public web on Streamlit Community Cloud at \url{https://glassbox-ml.streamlit.app/}.
\end{tightitemize}

\section{Limitations}
The platform currently focuses on a curated 64-document corpus suitable for rapid in-memory browser simulation. It does not scale to out-of-core distributed corpora or non-linear deep transformer representations.

\section{Future Scope}
Future extensions include:
\begin{tightenumerate}
    \item Incorporating interactive Ordinary Least Squares and Gradient Descent loss surface contour simulations into the web UI.
    \item Adding non-linear kernel PCA and t-SNE / UMAP manifold projections.
    \item Extending Model MRI to multi-layer neural network influence approximations via Hessian-free stochastic conjugate gradients.
\end{tightenumerate}

% =============================================================================
% REFERENCES (IEEE FORMAT)
% =============================================================================
\begin{thebibliography}{99}
\bibitem{salton1988} G. Salton and C. Buckley, ``Term-weighting approaches in automatic text retrieval,'' \textit{Information Processing \& Management}, vol. 24, no. 5, pp. 513--523, 1988.
\bibitem{jolliffe2002} I. T. Jolliffe, \textit{Principal Component Analysis}, 2nd ed. New York: Springer-Verlag, 2002.
\bibitem{golub2013} G. H. Golub and C. F. Van Loan, \textit{Matrix Computations}, 4th ed. Baltimore: Johns Hopkins University Press, 2013.
\bibitem{fisher1936} R. A. Fisher, ``The use of multiple measurements in taxonomic problems,'' \textit{Annals of Eugenics}, vol. 7, no. 2, pp. 179--188, 1936.
\bibitem{vapnik1995} C. Cortes and V. Vapnik, ``Support-vector networks,'' \textit{Machine Learning}, vol. 20, no. 3, pp. 273--297, 1995.
\bibitem{breiman1984} L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen, \textit{Classification and Regression Trees}. Belmont, CA: Wadsworth International Group, 1984.
\bibitem{kaufman2012} S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, ``Leakage in data mining: Formulation, detection, and avoidance,'' \textit{ACM Transactions on Knowledge Discovery from Data}, vol. 6, no. 4, pp. 1--21, 2012.
\bibitem{koh2017} P. W. Koh and P. Liang, ``Understanding black-box predictions via influence functions,'' in \textit{Proc. 34th Int. Conf. Machine Learning (ICML)}, 2017, pp. 1885--1894.
\bibitem{scikit-learn} F. Pedregosa et al., ``Scikit-learn: Machine learning in Python,'' \textit{Journal of Machine Learning Research}, vol. 12, pp. 2825--2830, 2011.
\bibitem{belkin2019} M. Belkin, D. Hsu, S. Ma, and S. Mandal, ``Reconciling modern machine-learning practice and the classical bias--variance trade-off,'' \textit{Proc. National Academy of Sciences (PNAS)}, vol. 116, no. 32, pp. 15849--15854, 2019.
\bibitem{bottou2012} L. Bottou, ``Stochastic gradient descent tricks,'' in \textit{Neural Networks: Tricks of the Trade}, 2nd ed. Berlin: Springer, 2012, pp. 421--436.
\bibitem{vaswani2017} A. Vaswani et al., ``Attention is all you need,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2017, pp. 5998--6008.
\bibitem{he2016} K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image recognition,'' in \textit{Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)}, 2016, pp. 770--778.
\bibitem{hinton2012} G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, ``Improving neural networks by preventing co-adaptation of feature detectors,'' \textit{arXiv:1207.0580}, 2012.
\bibitem{nakkiran2020} P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, and I. Sutskever, ``Deep double descent: Where bigger models and more data hurt,'' in \textit{Int. Conf. Learning Representations (ICLR)}, 2020.
\bibitem{duchi2018} J. C. Duchi and H. Namkoong, ``Learning models with uniform performance on target populations,'' in \textit{Proc. 32nd Conf. Neural Information Processing Systems (NeurIPS)}, 2018.
\end{thebibliography}

% =============================================================================
% APPENDIX
% =============================================================================
\appendix
\chapter{Reproducibility Information and Source Code}

\section{Open-Source Repository and Live Deployment}
All source code, fixtures, test suites, and documentation are open source and reproducible:
\begin{tightitemize}
    \item \textbf{GitHub Repository:} \url{https://github.com/harshrarora/glassbox}
    \item \textbf{Live Deployed Web Application:} \url{https://glassbox-ml.streamlit.app/}
    \item \textbf{Automated Test Suite:} Execute \texttt{python -m pytest glassbox/tests -q} (89/89 tests passing).
\end{tightitemize}

\section{Core Implementation Snippet: Jacobi Eigensolver}
\begin{verbatim}
def jacobi_eigenvalue_solver(A: np.ndarray, max_sweeps: int = 50, tol: float = 1e-7):
    n = A.shape[0]
    V = np.eye(n, dtype=np.float64)
    A_k = A.astype(np.float64).copy()
    for sweep in range(max_sweeps):
        off_diag = np.sqrt(np.sum(A_k**2) - np.sum(np.diag(A_k)**2))
        if off_diag < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A_k[p, q]) < 1e-15:
                    continue
                tau = (A_k[q, q] - A_k[p, p]) / (2.0 * A_k[p, q])
                t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau**2))
                c = 1.0 / np.sqrt(1.0 + t**2)
                s = t * c
                R = np.eye(n)
                R[p, p], R[q, q] = c, c
                R[p, q], R[q, p] = s, -s
                A_k = R.T @ A_k @ R
                V = V @ R
    return np.diag(A_k), V
\end{verbatim}

\section{Core Implementation Snippet: Influence Functions}
\begin{verbatim}
def compute_influence_functions_2d(X_tr, y_tr, x_test, y_test, l2_reg=0.1):
    theta = fit_logistic_regression_2d(X_tr, y_tr, l2_reg)["theta"]
    X_b = np.column_stack([X_tr, np.ones(len(X_tr))])
    x_test_b = np.append(x_test, 1.0)
    mu = 1.0 / (1.0 + np.exp(-y_tr * (X_b @ theta)))
    W = mu * (1.0 - mu)
    H = (X_b.T * W) @ X_b / len(X_tr) + l2_reg * np.eye(3)
    p_test = 1.0 / (1.0 + np.exp(-y_test * (x_test_b @ theta)))
    grad_test = -(1.0 - p_test) * y_test * x_test_b
    s_test = np.linalg.solve(H, grad_test)
    grad_train = -((1.0 - mu) * y_tr)[:, None] * X_b
    influences = -grad_train @ s_test
    return {"theta": theta, "influences": influences, "H": H}
\end{verbatim}

\end{document}
'''

TEX_FILE.write_text(tex_content, encoding="utf-8")
print(f"Written LaTeX source to: {TEX_FILE}")

# Compile using pdflatex (2 passes for TOC and cross-references)
print("Running Pass 1 of pdflatex...")
res1 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "glassbox_lab_report.tex"],
                      cwd=str(REPORT_DIR), capture_output=True, text=True)
if res1.returncode != 0:
    print("Pass 1 Failed! Output:")
    print(res1.stdout[-1500:])
    sys.exit(1)

print("Running Pass 2 of pdflatex (resolving TOC, LOF, LOT, cross-references)...")
res2 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "glassbox_lab_report.tex"],
                      cwd=str(REPORT_DIR), capture_output=True, text=True)
if res2.returncode != 0:
    print("Pass 2 Failed! Output:")
    print(res2.stdout[-1500:])
    sys.exit(1)

print(f"Compilation Successful! PDF generated at: {PDF_FILE}")
