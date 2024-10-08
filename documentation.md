# Project Documentation

## 1. Project Overview

### 1.1 Project Statement
- **Problem Statement**: Improper waste sorting at universities and public spaces leads to contamination of recyclable materials and reduces the efficiency of recycling programs. Users often misplace waste in the wrong bins due to confusion or lack of knowledge, which increases landfill waste and waste management costs.
- **Objectives**:
  - Develop an AI-powered waste recognition and sorting system to assist users in accurately disposing of their waste.
  - Build a system that can recognize different materials and suggest the appropriate recycling bin for each.
  - Eventually integrate the system into public waste management infrastructure for automation and scalability.
- **Target Audience**: 
  - University students and staff.
  - Public waste management systems.
  - Municipalities and industries involved in waste disposal and recycling.

### 1.2 Proposed Solution
- **Solution Overview**: The AI-powered waste recognition and sorting system will help users sort their waste accurately by using a scanning screen to identify the type of material in real-time. Users can scan their waste or drop it in a designated area, and the system will either provide instructions or automatically sort the waste into the correct bin.
- **Key Features**:
  - Real-time image recognition to classify waste into six categories (cardboard, glass, metal, paper, plastic, trash).
  - Standalone interface with scanning functionality, reducing the need for manual sorting.
  - Potential integration with automated sorting systems for public spaces or industrial settings.
- **Expected Impact**:
  - Increased recycling efficiency by reducing human errors in waste sorting.
  - Lower contamination rates of recyclable materials.
  - Cost savings for waste management organizations due to improved sorting processes.
  - Contributing to environmental sustainability and reducing landfill waste.

### 1.3 Technical Aspects
- **Programming Languages**:
  - **Python**: Used for model development, data processing, and machine learning.
  - **Torch (PyTorch)**: For building and training the neural network for image classification.
- **Frameworks and Libraries**:
  - **PyTorch**: Used for constructing and training the Convolutional Neural Network (CNN) for waste classification.
  - **Torchvision**: For image transformations and dataset handling.
  - **PIL (Python Imaging Library)**: For image preprocessing and loading during model testing.
  - **NumPy**: For handling array operations.
  - **AWS S3 and SageMaker**: AWS S3 was used for storing the 6790 labeled images dataset, while SageMaker Studio was employed for faster model training using better compute resources.

---

## 2. Documentation of AI Tools Used

### 2.1 Overview of AI Tools
- **Tool Name**: ChatGPT
- **Purpose**: ChatGPT was used for brainstorming project ideas, assisted learning, and debugging code during the development process. The team relied on it heavily as everything was new to them.
  
### 2.2 Application of AI Tools
- **When Applied**:
  - ChatGPT was applied throughout the project, especially during the initial stages for brainstorming and learning AWS services.
- **How Applied**:
  - The team used ChatGPT for generating project ideas, understanding AWS services, and debugging code when they encountered issues.
- **Rationale**:
  - ChatGPT was chosen for its versatility in providing quick, reliable assistance across different phases of the project, especially as the team was new to AI and cloud services.

### 2.3 Total use of AI Tools

#### Participant 1 -- Patrick Agnimel
- **% Usage**:
  - **Brainstorming**: Patrick used ChatGPT to generate the initial idea for the project and research AWS services to use.
  - **Assisted Learning**: Regular use of ChatGPT for understanding AI tools and concepts.
  - **Debugging**: Used ChatGPT to resolve issues while writing and testing code.

#### Participant 2 -- Devin Huang
- **% Usage**:
  - **Optimization**: optimized the model and added multiple QOL changes to the code, chat GPT was used partially for the optimization and QOL
  - **Debugging**: Fixed some bugs/ errors that we added to the code.
  - **80% LLM usage**: LLM was used a lot to help me understand different machine learning related concepts, additionally it was used for setting up AWS

#### Participant 3 -- Nikola Rowe
- **% Usage**:
  - **brainstorming**: Used ChatGPT to understand the voted idea and to research on each step
  - **Development**: Used ChatGPT to understand the initial process of training and creating a CNN model, and used to debug and fix while working on model.


#### Participant 4 -- fahad aljahlan
- **% Usage**:: used chatgpt to help reverte back to the correct version and merge it to the main, only lost the doc
- merged the data sets to train the ai and test it, worked on aws but went with my teamates code because it was better

#### Participant 5 -- name
- **% Usage**: Detail the percentage of AI tool usage by the participant across different project phases, such as brainstorming, development, documentation, and testing. Specify which tools were used most frequently.
