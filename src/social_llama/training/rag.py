"""Retrieval-Augmented Generation (RAG) system for classification."""
import ast
import logging
import os

import datasets
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from social_llama.config import DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS
from social_llama.config import DATA_DIR_EVALUATION_SOCKET
from social_llama.config import DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED
from social_llama.config import DATA_DIR_VECTOR_DB
from social_llama.evaluation.helper_functions import label_finder
from social_llama.utils import save_json


load_dotenv()


class RAGClassification:
    """RAG Based classificaiton system."""

    def __init__(
        self,
        model_name: str,
        model_name_embedding: str = "sentence-transformers/all-MiniLM-l6-v2",
    ) -> None:
        """Initializes the RAG classification system."""
        self.model_name = model_name
        self.model_name_embedding = model_name_embedding
        self.model_kwargs: dict = {"device": self._get_device()}
        self.encode_kwargs: dict = {"normalize_embeddings": True}

    def convert_data_to_langchain(self, dataset, is_socket: bool = False):
        """Converts a HuggingFace dataset to a list of langchain documents.

        Args:
            dataset (datasets.Dataset): HuggingFace dataset.

        Returns:
            docs (list): List of langchain documents.
        """
        docs = []
        if is_socket:
            for idx, d in enumerate(DataLoader(dataset["train"])):
                docs.append(
                    Document(
                        page_content=d["text"][0],
                        metadata={
                            "idx": idx,
                            "label": labels_mapping[d["label"].item()],
                        },
                    )
                )
        else:
            for d in DataLoader(dataset["train"]):
                docs.append(
                    Document(
                        page_content=d["text"][0],
                        metadata={
                            "idx": d["idx"].item(),
                            "label": d["response_good"][0],
                        },
                    )
                )
        return docs

    def make_or_load_vector_db(
        self,
        dataset_name: str,
        data: list,
        remake_db: bool = False,
    ):
        """Creates or loads a vector database.

        Args:
            dataset_name (str): Name of the dataset.
            data (list): List of langchain documents.
            model_name_embedding (str): Name of the pre-trained model to use for the vector database.
            model_kwargs (dict): Model configuration options.
            encode_kwargs (dict): Encoding options.
            remake_db (bool): Whether to remake the vector database.

        Returns:
            db (FAISS): Vector database.
            retriever (FAISSRetriever): Vector database retriever.
        """
        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name_embedding,  # Provide the pre-trained model's path
            model_kwargs=self.model_kwargs,  # Pass the model configuration options
            encode_kwargs=self.encode_kwargs,  # Pass the encoding options
        )

        # Check if there exist a vector database with a name
        if (
            os.path.exists(str(DATA_DIR_VECTOR_DB / f"{dataset_name}.faiss"))
            and not remake_db
        ):
            logging.info(f"Vector database {dataset_name}.faiss exists. Loading...")
            db = FAISS.load_local(
                str(DATA_DIR_VECTOR_DB / f"{dataset_name}.faiss"), embeddings
            )

        else:
            logging.info(
                f"Vector database {dataset_name}.faiss does not exist. Creating..."
            )
            # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
            # It splits text into chunks of 1000 characters each with a 150-character overlap.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )

            # 'data' holds the text you want to split, split the text into documents using the text splitter.
            docs = text_splitter.split_documents(data)

            db = FAISS.from_documents(docs, embeddings)

            # Change distance strategy to cosine similarity
            db.distance_strategy = DistanceStrategy.COSINE

            # Save the vector database to the specified path
            db.save_local(str(DATA_DIR_VECTOR_DB / f"{dataset_name}.faiss"))
            logging.info(f"Vector database {dataset_name}.faiss created and saved.")

        retriever = db.as_retriever(search_kwargs={"k": 5})

        return db, retriever

    def _get_device(self):
        """Gets the device to use for the model.

        Returns:
            device (torch.device): Device to use for the model.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # Find the device type # include looking for mps
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logging.info(f"Using device: {device}")
        return device

    def decode_documents(self, response):
        """Decodes the documents returned by the vector database.

        Args:
            response (list): List of tuples containing the document and the score.

        Returns:
            formatted_response (str): Formatted response.
        """
        formatted_response = []
        for idx, (doc, _) in enumerate(response):
            content = doc.page_content
            label = doc.metadata["label"]
            formatted_response.append(
                f'Document {idx+1}: "{content}"\nLabel {idx}: {label}'
            )
        return "\n".join(formatted_response)


class HuggingfaceChatTemplate:
    """Huggingface chat template for RAG."""

    def __init__(self, model_name: str):
        """Initializes the Huggingface chat template.

        Args:
            model_name (str): Name of the pre-trained model.

        Returns:
            None
        """
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.use_default_system_prompt = False

    def get_template_classification(self, system_prompt: str, task: str) -> str:
        """Gets the template for classification.

        Args:
            system_prompt (str): System prompt.
            task (str): Task.

        Returns:
            template (str): Template.
        """
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nRetrieved Documents:\n{context}\nInput Text: {text}\nAnswer:""".format(
                    task=task,
                    context="{context}",
                    text="{text}",
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False)


# Load the data
dataset_names = ["social-dimensions"]
dataset_names = [
    # "contextual-abuse#PersonDirectedAbuse",
    # "contextual-abuse#IdentityDirectedAbuse",
    # "tweet_irony",
    # "hateoffensive",
    # "tweet_emotion",
    # "implicit-hate#explicit_hate",
    # "implicit-hate#implicit_hate",
    # "crowdflower",
    # "dailydialog",
    "hahackathon#is_humor",
    "sarc",
    "questionintimacy",
]

for dataset_name in dataset_names:
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        filename=f"logs/rag_{dataset_name}.log",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    logging.info("Loading training and test datasets.")

    if dataset_name == "social-dimensions":
        dataset = datasets.load_dataset(
            "json", data_files=str(DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED / "train.json")
        )
        dataset_test = datasets.load_dataset(
            "json", data_files=str(DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED / "test.json")
        )
        labels = [
            "social_support",
            "conflict",
            "trust",
            "fun",
            "similarity",
            "identity",
            "respect",
            "romance",
            "knowledge",
            "power",
            "other",
        ]
        label_descriptions = """social_support: Giving emotional or practical aid and companionship.
    conflict: Contrast or diverging views.
    trust: Will of relying on the actions or judgments of another.
    fun: Experiencing leisure, laughter, and joy.
    similarity: Shared interests, motivations or outlooks.
    identity: Shared sense of belonging to the same community or group.
    respect: Conferring status, respect, appreciation, gratitude, or admiration upon another.
    romance: Intimacy among people with a sentimental or sexual relationship.
    knowledge: Exchange of ideas or information; learning, teaching.
    power: Having power over the behavior and outcomes of another.
    other: If none of the above social dimensions apply."""
        is_socket = False

    else:
        dataset = datasets.load_dataset(
            "Blablablab/SOCKET", dataset_name, split="train"
        )
        dataset_test = datasets.load_dataset(
            "Blablablab/SOCKET", dataset_name, split="test"
        )
        # if length is more than 2000, randomly sample 2000
        if len(dataset_test) > 2000:
            dataset_test = dataset_test.shuffle(seed=42).select(range(2000))
        # Convert to DatasetDict for consistency
        dataset = datasets.DatasetDict(
            {
                "train": dataset,
            }
        )
        dataset_test = datasets.DatasetDict(
            {
                "train": dataset_test,
            }
        )
        # labels: List[str] = dataset['train'].features["label"].names
        # labels_mapping: Dict[int, str] = {i: label for i, label in enumerate(labels)}
        socket_prompts: pd.DataFrame = pd.read_csv(
            DATA_DIR_EVALUATION_SOCKET / "socket_prompts_knowledge.csv"
        )
        label_descriptions = socket_prompts[socket_prompts["task"] == dataset_name][
            "knowledge"
        ].iloc[0]
        labels = ast.literal_eval(
            socket_prompts[socket_prompts["task"] == dataset_name]["options"].iloc[0]
        )
        labels_mapping = {i: label for i, label in enumerate(labels)}
        is_socket = True

    # Specify the model name you want to use
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    RAG = RAGClassification(
        model_name=model_name,
        model_name_embedding="sentence-transformers/all-MiniLM-l6-v2",
    )

    # Convert to langchain format
    docs = RAG.convert_data_to_langchain(dataset, is_socket=is_socket)
    docs_test = RAG.convert_data_to_langchain(dataset_test, is_socket=is_socket)

    # Make or load vector db
    logging.info(f"Making or loading vector database for '{dataset_name}'.")
    db, retriever = RAG.make_or_load_vector_db(
        dataset_name,
        docs,
        remake_db=True,
    )

    llm = InferenceClient(
        model=model_name,
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )
    # Disable caching
    llm.headers["x-use-cache"] = "0"

    system_prompt = """You are part of a RAG classification system designed to categorize texts.
    Your task is to analyze the input text and classify it into one of the provided labels based on your general knowledge and the context provided by any retrieved documents that may be relevant.
    Below are the labels you can choose from, along with their descriptions. Use the information from the retrieved documents to aid your decision if they are relevant to the input text.

    Labels and Descriptions:
    {label_descriptions}
    """.format(
        label_descriptions=label_descriptions
    )

    task = """Using the general knowledge and the information from the retrieved documents provided above, classify the input text by selecting the most appropriate label.
    Consider the relevance and content of each document in relation to the input text and the descriptions of the labels.
    If a retrieved document is highly relevant to the input text and aligns closely with the description of a label, that label might be the correct classification.
    """

    template = HuggingfaceChatTemplate(
        model_name=model_name,
    ).get_template_classification(
        system_prompt=system_prompt,
        task=task,
    )

    # Group by idx and collect labels
    test_data_formatted = {}

    # Loop through each JSON object and group by 'idx'
    for id_, obj in enumerate(DataLoader(dataset_test["train"])):
        idx = obj["idx"].item() if dataset_name == "social-dimensions" else id_
        response_good = (
            obj["response_good"][0]
            if dataset_name == "social-dimensions"
            else obj["label"]
        )

        if idx not in test_data_formatted:
            test_data_formatted[idx] = {
                "label": []
                if dataset_name == "social-dimensions"
                else labels_mapping[response_good.item()],
                "idx": idx,
                "text": obj["text"][0]
                if dataset_name == "social-dimensions"
                else obj["text"][0],
            }
        else:
            test_data_formatted[idx]["label"].append(response_good)

    # Return a list of all the values in the dictionary
    test_data_formatted = list(test_data_formatted.values())

    predictions = []

    for idx, sample in tqdm(enumerate(test_data_formatted), desc="Predicting"):
        search_docs_text = db.similarity_search_with_score(
            sample["text"], k=5, fetch_k=10
        )
        # searchDocs_question = db.similarity_search_with_score(question, k=5, fetch_k=25)

        decoded_text = RAG.decode_documents(search_docs_text)
        # decoded_question = decode_documents(searchDocs_question)

        has_output = False

        # This is need as the LLM client sometimes is just hanging and needs to be reinitialized
        while not has_output:
            try:
                output = llm.text_generation(
                    template.format(
                        context=decoded_text,
                        text=sample["text"],
                    ),
                    max_new_tokens=150,
                    temperature=0.7,
                    # repetition_penalty=1.2,
                )
                has_output = True

            except Exception as e:
                logging.info(f"Error: {e}")

                # Delete LLM
                del llm

                logging.info("Reinitializing LLM...")
                llm = InferenceClient(
                    model=model_name,
                    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                )
                # Disable caching
                llm.headers["x-use-cache"] = "0"

        label = label_finder(output, labels)

        predictions.append(
            {
                "idx": idx,
                "text": sample["text"],
                "label": sample["label"],
                "prediction": label,
                "output": output,
                "documents": decoded_text,
            }
        )

    if is_socket:
        save_path = (
            DATA_DIR_EVALUATION_SOCKET
            / f"{dataset_name}/{model_name}_predictions_RAG.json"
        )
    else:
        save_path = (
            DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS / f"{model_name}_predictions_RAG.json"
        )

    # Save predictions to JSON file
    save_json(save_path, predictions)
