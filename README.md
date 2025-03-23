# Name: Swaraj Bhanja | Student Id: st125052

# Welcome to Model Distillation!

This is a web-based end-to-end application named Model Distillation. It leverages the power of deep learning and web development to provide a website that performs text toxicity classification based on the input sentence or phrase.

# About the Deep Learning Model

The brains of this solution is the deep learning model trained for this purpose. The DL model was trained based on the **BERT** pre-trained model being subject to distillation from a teacher model to a student model and LoRA adapters. **BERT** is pretrained on a large corpus (like Wikipedia) using tasks like Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). After pretraining, it can be fine-tuned for specific NLP tasks such as text classification, question answering, sentiment analysis, and more, often achieving state-of-the-art performance.

# Loading the Hate Speech Dataset

The datasets library from Hugging Face is imported, which is used for accessing and managing machine learning datasets. A mapping `task_to_keys` is set up to define the input field for the task, identifying "text" as the key feature. The dataset named `hate_speech18` is then loaded using Hugging Face's `load_dataset` function, pulling the training portion of the dataset. The dataset is split into three parts: 80% is kept for training, and the remaining 20% is equally divided into validation and test sets, ensuring reproducibility using a fixed `SEED`. Finally, all three splits are combined into a DatasetDict object called `raw_datasets`, which conveniently groups the train, validation, and test sets for future use in model training and evaluation.

A `label_list` is created, which contains all the unique labels used in the dataset `("noHate", "hate", "idk/skip", "relation")`. Then, a dictionary called `label2id` is created that assigns each label a unique numerical ID—this is important because machine learning models work with numbers, not text. For instance, "hate speech" might be mapped to 0, "neutral" to 1, and so on. The next line reverses this mapping to create id2label, which allows you to convert back from a numerical prediction to a human-readable label. These two dictionaries will be helpful when training the model and interpreting its predictions.

## Loading the Pre-Trained BERT Teacher Model

The count of unique labels exist in the training data—this value, `num_labels`, which tells the model how many output categories it needs to predict. Then, using Hugging Face's transformers library, a pretrained BERT model called `"bert-base-uncased` is loaded. The `AutoTokenizer` is loaded to break down input sentences into tokens that BERT can understand. After that, the actual model `(teacher_model)` is loaded and adapted for a classification task by specifying how many output labels it should have, as well as providing the mappings between labels and IDs `(label2id and id2label)` so that it can make sense of predictions and training targets. 

## Tokenization

 A `tokenize_function` is defined that takes each example and uses the `tokenizer` (loaded earlier) to break the text into tokens while ensuring a consistent maximum length of 128 words, trimming anything beyond that. This function is applied across the entire dataset using `.map()` with batching enabled for speed. After tokenization, the dataset is cleaned up by removing unnecessary columns like `user_id`, `subforum_id`, and `num_contexts`. The label column is then renamed from `label` to `labels` to match the expected input for Hugging Face models. Finally, the dataset is formatted to use PyTorch tensors, so it’s ready to be fed into the model for training or evaluation. In essence, this snippet transforms the raw textual data into clean, tokenized, and model-ready input.

## Data Collation

A `DataCollatorWithPadding` is generated, which ensures that in each batch, all input sequences are padded to the length of the longest sequence—this dynamic padding is more efficient than always padding to a fixed length. Then, the training, validation, and test datasets are shuffled to randomize the order of data points using a fixed `SEED` for reproducibility. Finally, each of these datasets is wrapped into a `DataLoader, which handles batching and shuffling (only for training) and ensures that the data collator takes care of padding during loading. The training `DataLoader` is set to shuffle the data to avoid learning biases from the order, and all DataLoaders are set to process 32 examples at a time. In simple terms, this snippet prepares the data in mini-packets (batches) and adds automatic padding, making it ready for efficient model training and evaluation using PyTorch.

## Student Model Creation

A lighter, more compact version of the original BERT model, known as the `student` model is created, which is typically used in model distillation to mimic the performance of a larger `teacher` model while being more efficient. Yhe configuration of the already-loaded teacher model is extracted (which includes details like number of layers, hidden units, etc.) as a Python dictionary. Then, the number of hidden layers is reduced by half, essentially shrinking the depth of the network to make it smaller and faster. This modified configuration is then converted back into a formal BERT configuration object `(BertConfig)`, which is used to initialize a new model of the same type as the teacher (here, a sequence classification BERT model), but with this smaller setup. Importantly, this new model is uninitialized, meaning its weights haven't been trained yet. 

## Student Model Distillation Logic

A custom function, `distill_bert_weights`, is created that transfers knowledge from a larger, pretrained BERT model (the `teacher`) to a smaller, more efficient version (the `student`). It works by recursively navigating through the internal components of both models and copying over weights. If the part being compared is a full BERT model or classification head (detected using `isinstance` checks), the function walks through each submodule and applies itself recursively. The special case is the BERT encoder, the core part of the model where most of the deep learning happens. Since the student has fewer layers, only every second layer from the teacher is copied (either odd or even depending on `2*i` or `2*i+1`), thereby trying to preserve the depthwise essence of learning while reducing size. If the component is not an encoder or a full model (like a classification head), the weights are directly copied using `load_state_dict`. 

## Student Model LoRA Logic

LoRA is set up `(Low-Rank Adaptation)` on the student model to make fine-tuning more efficient and lightweight. LoRA is a technique that adds a small number of trainable parameters (called adapters) to a large model, allowing it to be adapted to new tasks without updating all its original weights. The function `get_lora_model` takes in the student model and creates a LoraConfig specifying key parameters like `r=8` (which defines the rank or size of the adaptation matrices), `lora_alpha=16` (a scaling factor), and a `dropout of 0.1` to prevent overfitting. The task type is set to `SEQ_CLS`, indicating a sequence classification problem like hate speech detection. `inference_mode=False` means the model will be trained, not just used for predictions. The `get_peft_model` function then wraps the student model with these LoRA adapters. 

# Initializing Training Arguments

A learning rate (lr) of 5e-5 and the Adam optimizer, which is a widely used optimization algorithm that adjusts model weights to minimize loss are defined. The optimizer is configured to update all the parameters in the student model. Both the student and teacher models are then moved to the designated computing device (either a GPU or CPU) to ensure computations run efficiently. Next, the code uses Hugging Face’s `get_scheduler` to create a linear learning rate scheduler, which gradually decreases the learning rate over time from its initial value to zero. This helps the model converge more smoothly during training. The scheduler is set with no warm-up steps (meaning it starts reducing the learning rate from the beginning), and the total number of training steps is calculated by multiplying the number of epochs by the number of batches per epoch.

# Training using Distillation

At the start, tracking tools like progress_bar for visual feedback, and lists to log different types of losses over time: classification loss, distillation divergence loss, and cosine similarity loss are initializaed. Inside the loop, for each epoch, the student model is put into training mode, while the teacher remains in evaluation mode (to freeze its weights). For each training batch, outputs from both the student and teacher are computed. Then three loss components: the standard classification loss from the student, a divergence loss comparing student and teacher logits, and a cosine loss that encourages similar prediction directions between student and teacher are calculated. These three losses are averaged to get the final loss, which is used to update the model’s weights via backpropagation and an optimizer step. After training, the model is evaluated: it predicts labels on the validation set, computes classification loss, and tracks accuracy using a metric. The accuracy and loss values are printed and stored after each epoch, and at the end, the average accuracy across all epochs is displayed. In short, this loop fine-tunes the student model using a combination of standard and distillation-based objectives, while monitoring its performance throughout.

# Training using LoRA

The LoRA version of the training loop removes the distillation aspect and focuses solely on training the student model independently, without involving the teacher's predictions. As before, it tracks various losses and uses tqdm for a progress bar. Inside each epoch, the student model is trained by computing the standard classification loss `(loss_cls)`, which measures how far the student’s predictions are from the true labels. The distillation loss `(loss_div)` and cosine similarity loss `(loss_cos)` are set to zero manually, effectively turning off knowledge distillation and focusing purely on supervised learning. Since the distillation losses are zero, the final loss is just the classification loss. The loss is backpropagated to update the model's weights, and the learning rate scheduler is stepped accordingly. After each epoch, the model is evaluated on the validation set, where it predicts labels and computes accuracy using a metric tracker. The classification and evaluation losses are logged for analysis. 

## Testing

A function, `predict_toxicity`, is designed to make predictions on a single piece of text using a trained classification model, specifically to assess whether the input is toxic or not. It first uses the tokenizer to convert the raw input text into a format the model understands, applying truncation and padding to handle different text lengths. The tokenized inputs are then moved to the same device (CPU or GPU) as the model. Inside a `torch.no_grad()` block (which disables gradient computation for efficiency), the inputs are passed through the model to get raw output scores, called logits. These logits are then transformed into probabilities using the softmax function. The class with the highest probability is selected as the predicted label, and its human-readable name is looked up using `id2word`. Finally, the function returns both the predicted class label (like `toxic` or `not toxic`) and the associated `confidence` score. 

[Analysis Metrics](https://github.com/st125052/a5-nlp-direct-preference-optimization-st125052/blob/main/notebooks/pdfs/Training%20Metrics%20Based%20on%20Hyperparameter%20Experimentation.pdf)


# Website Creation
The model was then hosted over the Internet with Flask as the backend, HTML, CSS, JS as the front end, and Docker as the container. The end-user is presented with a UI wherein a search input box is present. Once the user types in the first set of words, they click on the `Get Prediction` button. The input texts are sent to the JS handler which makes an API call to the Flask backend. The Flask backend has the GET route which intercepts the HTTP request. The result is then returned back to the JS handler as a list by the Flask backend. The JS handler then maps the response to the result container's inner HTML and finally makes it visible for the output to be shown. 

A Vanilla architecture was chosen due to time constraints. In a more professional scenario, the ideal approach would be used frameworks like React, Angular and Vue for Frontend and ASP.NET with Flask or Django for Backend.

The following describes the key points of the hosting discussion.
> **1. DigitalOcean (Hosting Provider)**
> 
>> - **Role:** Hosting and Server Management
>> - **Droplet:** Hosts the website on a virtual server, where all files, databases, and applications reside.
>> - **Dockerized Container:** The website is hosted in a Dockerized container running on the droplet. The container is built over a Ubuntu Linux 24.10 image.
>> - **Ports and Flask App:** The Dockerized container is configured to host the website on port 8000. It forwards requests to port 5000, where the Flask app serves the backend and static files. This flask app contains the pickled model, which is used for prediction.
>> - **IP Address:** The droplet’s public IP address directs traffic to the server.
>
>  **In Summary:** DigitalOcean is responsible for hosting the website within a Dockerized container, ensuring it is online and accessible via its IP address.
> 
>  **2. GoDaddy (Domain Registrar)**
>
>> - **Role:** Domain Registration and Management
>> - **Domain Purchase:** Registers and manages the domain name.
>> - **DNS Management:** Initially provided DNS setup, allowing the domain to be pointed to the DigitalOcean droplet’s IP address.
> 
> **In Summary:** GoDaddy ensures the domain name is registered and correctly points to the website’s hosting server.
>
>  **3. Cloudflare (DNS and Security/Performance Optimization)**
>
>> - **Role:** DNS Management, Security, and Performance Optimization
>> - **DNS Management:** Resolves the domain to the correct IP address, directing traffic to the DigitalOcean droplet.
>> - **CDN and Security:** Caches website content globally, enhances performance, and provides security features like DDoS protection and SSL encryption.
> 
> **In Summary:** Cloudflare improves the website’s speed, security, and reliability.
>
> **How It Works Together:**
> 
>> - **Domain Resolution:** The domain is registered with GoDaddy, which points it to Cloudflare's DNS servers. Cloudflare resolves the domain to the DigitalOcean droplet's IP address.
>> - **Content Delivery:** Cloudflare may serve cached content or forward requests to DigitalOcean, which processes and serves the website content to users.
> 
> **Advantages of This Setup:**
>
>> - **Security:** Cloudflare provides DDoS protection, SSL/TLS encryption, and a web application firewall.
>> - **Performance:** Cloudflare’s CDN reduces load times by caching content globally, while DigitalOcean offers scalable hosting resources.
>> - **Reliability:** The combination of GoDaddy, Cloudflare, and DigitalOcean ensures the website is always accessible, with optimized DNS resolution and robust hosting.

# Demo
https://github.com/user-attachments/assets/710718d0-a902-4b07-a919-91de71e7d7a3

# Access The Final Website
You can access the website [here](https://aitmltask.online). 

# Limitations
Note that the model predicts the toxicity correctly only for a small handful texts or phrases. This could be attributed as a limitation of the hate speech dataset not having sufficent variety of examples, which is a known limitation.

# How to Run the Model Distillation Docker Container Locally
### Step 1: Clone the Repository
> - First, clone the repository to your local machine.
### Step 2: Install Docker
> - If you don't have Docker installed, you can download and install it from the [Docker](https://www.docker.com) website.
### Step 3: Build and Run the Docker Container
Once Docker is installed, navigate to the app folder in the project directory. Delete the docker-compose-deployment.yml file and run the following commands to build and run the Docker container:
> - `docker compose up -d`

### Important Notes
> - The above commands will serve the Docker container on port 5000 and forward the requests to the Flask application running on port 5000 in the containerized environment.
> - Ensure Ports Are Free: Make sure that port 5000 is not already in use on your machine before running the container.
> - Changing Flask's Port: If you wish to change the port Flask runs on (currently set to 5000), you must update the port in the app.py file. After making the change, remember to rebuild the Docker image in the next step. Execute the following command to stop the process: `docker compose down`. Then goto Docker Desktop and delete the container and image from docker. 
