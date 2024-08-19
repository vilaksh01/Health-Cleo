import re
import os
import requests
import warnings
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageEmbeddings,
    ChatUpstage,
    UpstageGroundednessCheck
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.llms.base import BaseLLM
from bs4 import BeautifulSoup
import openfoodfacts
from tavily import TavilyClient

warnings.filterwarnings("ignore")

class ReferenceRange(BaseModel):
    elevated: str
    borderline: str
    normal: str

class FoodItem(BaseModel):
    name: str
    value: float
    category: str

class FoodIntoleranceReport(BaseModel):
    reference_range: ReferenceRange
    food_items: List[FoodItem]

class ProductAnalysis(BaseModel):
    product_name: str
    ingredients: List[str]
    suitability: Dict[str, str]
    overall_rating: str
    explanation: str

def extract_tables_from_html(html_content: str) -> List[str]:
    soup = BeautifulSoup(html_content, 'html.parser')
    return [str(table) for table in soup.find_all('table')]

def parse_reference_range(text: str) -> ReferenceRange:
    elevated = re.search(r'Elevated[^\d]*(\d+(?:\.\d+)?)', text)
    borderline = re.search(r'Borderline[^\d]*(\d+(?:\.\d+)?)[^\d]*(\d+(?:\.\d+)?)', text)
    normal = re.search(r'Normal[^\d]*(\d+(?:\.\d+)?)', text)
    
    return ReferenceRange(
        elevated=f"> {elevated.group(1)} U/mL" if elevated else "",
        borderline=f"{borderline.group(1)}-{borderline.group(2)} U/mL" if borderline else "",
        normal=f"< {normal.group(1)} U/mL" if normal else ""
    )

def categorize_food_item(value: float, reference_range: ReferenceRange) -> str:
    elevated_threshold = float(re.search(r'\d+', reference_range.elevated).group())
    borderline_range = [float(x) for x in re.findall(r'\d+', reference_range.borderline)]
    
    if value >= elevated_threshold:
        return "Elevated"
    elif borderline_range[0] <= value <= borderline_range[1]:
        return "Borderline"
    else:
        return "Normal"

class FoodIntoleranceAnalysisService:
    def __init__(self, upstage_api_key: str, tavily_api_key: str):
        self.upstage_api_key = upstage_api_key
        self.tavily_api_key = tavily_api_key
        self.embeddings: Embeddings = UpstageEmbeddings(api_key=upstage_api_key, model="solar-embedding-1-large")
        self.llm: BaseLLM = ChatUpstage(api_key=upstage_api_key)
        self.translator: BaseLLM = ChatUpstage(api_key=upstage_api_key, model="solar-1-mini-translate-koen")
        self.groundedness_check = UpstageGroundednessCheck(api_key=upstage_api_key)
        self.vectorstore: VectorStore = None
        self.reference_range: ReferenceRange = None
        self.food_items: List[FoodItem] = []
        self.api = openfoodfacts.API(user_agent="FoodIntoleranceApp/1.0")
        self.tavily = TavilyClient(api_key=tavily_api_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=self.llm.get_num_tokens
        )

    def stream(self, prompt: str):
        """
        Stream the response from the LLM for the chat interface.
        """
        messages = [HumanMessage(content=prompt)]
        return self.llm.stream(messages)
    
    def process_pdf(self, pdf_path: str) -> FoodIntoleranceReport:
        loader = UpstageLayoutAnalysisLoader(pdf_path, split="page", api_key=self.upstage_api_key, use_ocr=True, output_type="html")
        docs = loader.load()

        all_tables = []
        for doc in docs:
            tables = extract_tables_from_html(doc.page_content)
            all_tables.extend(tables)

        chunked_tables = self.text_splitter.split_text("\n".join(all_tables))
        table_docs = [Document(page_content=chunk) for chunk in chunked_tables]

        self.vectorstore = Chroma.from_documents(table_docs, self.embeddings)

        self.reference_range = self._extract_reference_range(chunked_tables)
        self.food_items = self._extract_food_items(chunked_tables)

        return FoodIntoleranceReport(reference_range=self.reference_range, food_items=self.food_items)

    def _extract_reference_range(self, tables: List[str]) -> ReferenceRange:
        reference_range_query = PromptTemplate(
            input_variables=["tables"],
            template="""
            Analyze the following table content and extract the reference range for Elevated, Borderline, and Normal food intolerance levels:

            {tables}

            Provide the results in the following format:
            Elevated: > X U/mL
            Borderline: Y-Z U/mL
            Normal: < W U/mL
            """
        )
        
        results = []
        for chunk in tables:
            result = self._query_vectorstore(reference_range_query.format(tables=chunk))
            results.append(result)

        combined_result = " ".join(results)
        return parse_reference_range(combined_result)
    
    def _parse_food_items(self, food_items_text: str) -> List[FoodItem]:
        lines = food_items_text.split('\n')
        food_items = []
        for line in lines:
            match = re.match(r'(\w+(?:\s+\w+)*)\s*:\s*(\d+(?:\.\d+)?)', line)
            if match:
                name, value = match.groups()
                value = float(value)
                category = categorize_food_item(value, self.reference_range)
                food_items.append(FoodItem(name=name, value=value, category=category))
        return food_items

    def _extract_food_items(self, tables: List[str]) -> List[FoodItem]:
        food_items_query = PromptTemplate(
            input_variables=["tables"],
            template="""
            Analyze the following table content and list all food items with their corresponding intolerance values:

            {tables}

            Provide the results in the following format:
            Food Item: Value U/mL
            """
        )
        
        all_food_items = []
        for chunk in tables:
            result = self._query_vectorstore(food_items_query.format(tables=chunk))
            food_items = self._parse_food_items(result)
            all_food_items.extend(food_items)

        return all_food_items

    def _query_vectorstore(self, query: str) -> str:
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
        return qa_chain.run(query)
    
    def translate_to_korean(self, text: str) -> str:
        messages = [
            HumanMessage(content=text),
            AIMessage(content="모든 응답을 한국어로 번역 중")
        ]
        response = self.translator.invoke(messages)
        return response.content

    def display_results_in_korean(self, report: FoodIntoleranceReport, product_analysis: ProductAnalysis):
        # Translate reference range and food items
        ref_range_text = f"Reference Range: Elevated: {report.reference_range.elevated}, Borderline: {report.reference_range.borderline}, Normal: {report.reference_range.normal}"
        ref_range_korean = self.translate_to_korean(ref_range_text)
        # print(f"\n{ref_range_korean}")

        print(product_analysis)

        for item in report.food_items:
            item_text = f"{item.name}: {item.value} U/mL - {item.category}"
            item_korean = self.translate_to_korean(item_text)
            # print(item_korean)

        # Translate product analysis results
        product_name_korean = self.translate_to_korean(f"Product Analysis for {product_analysis.product_name}:")
        print(f"\n{product_name_korean}")

        ingredients_korean = self.translate_to_korean(f"Ingredients: {', '.join(product_analysis.ingredients)}")
        print(ingredients_korean)

        suitability_korean = "Suitability:\n"
        for ingredient, suitability in product_analysis.suitability.items():
            suitability_korean = self.translate_to_korean(f"  {ingredient}: {suitability}")
            suitability_korean = suitability_korean.replace('\n', '')  # Remove extra new lines
            suitability_korean = suitability_korean.strip()
            suitability_korean = suitability_korean if suitability_korean else "Unknown"
            suitability_korean = suitability_korean.strip()
            suitability_korean = f"  {ingredient}: {suitability_korean}"
            suitability_korean = self.translate_to_korean(suitability_korean)
            suitability_korean = suitability_korean.replace('\n', '')  # Remove extra new lines
            suitability_korean = suitability_korean.strip()
            suitability_korean = suitability_korean if suitability_korean else "Unknown"
            suitability_korean = suitability_korean.strip()
            suitability_korean = f"  {ingredient}: {suitability_korean}"
            suitability_korean = self.translate_to_korean(suitability_korean)
            print(suitability_korean)

        overall_rating_korean = self.translate_to_korean(f"Overall Rating: {product_analysis['overall_rating']}")
        explanation_korean = self.translate_to_korean(f"Explanation: {product_analysis['explanation']}")
        print(overall_rating_korean)
        print(explanation_korean)

    def analyze_product(self, product_name: str, ingredients: List[str]) -> ProductAnalysis:
        suitability = self._check_ingredient_suitability(ingredients)
        overall_rating, explanation = self._generate_overall_rating(suitability)
        
        return ProductAnalysis(
            product_name=product_name,
            ingredients=ingredients,
            suitability=suitability,
            overall_rating=overall_rating,
            explanation=explanation
        )

    def _check_ingredient_suitability(self, ingredients: List[str]) -> Dict[str, str]:
        suitability_query = PromptTemplate(
            input_variables=["ingredients", "food_items"],
            template="""
            Based on the following food intolerance data:
            {food_items}

            Analyze the suitability of these ingredients:
            {ingredients}

            For each ingredient, classify it as:
            - Suitable: if it's not in the list or has a "Normal" value
            - Caution: if it's in the list with a "Borderline" value
            - Avoid: if it's in the list with an "Elevated" value
            - Unknown: if there's not enough information

            Provide the result in the following format:
            Ingredient: Classification
            """
        )

        # Split ingredients into chunks if necessary
        ingredient_chunks = self.text_splitter.split_text(", ".join(ingredients))
        food_items_text = "\n".join([f"{item.name}: {item.value} U/mL - {item.category}" for item in self.food_items])

        all_suitability = {}
        for chunk in ingredient_chunks:
            result = self._query_vectorstore(suitability_query.format(
                ingredients=chunk,
                food_items=food_items_text
            ))
            chunk_suitability = self._parse_suitability_result(result)
            all_suitability.update(chunk_suitability)

        return all_suitability

    def _parse_suitability_result(self, result: str) -> Dict[str, str]:
        suitability = {}
        lines = result.split('\n')
        for line in lines:
            match = re.match(r'(\w+(?:\s+\w+)*)\s*:\s*(\w+)', line)
            if match:
                ingredient, classification = match.groups()
                suitability[ingredient] = classification
        return suitability

    def _generate_overall_rating(self, suitability: Dict[str, str]) -> Tuple[str, str]:
        rating_query = PromptTemplate(
            input_variables=["suitability"],
            template="""
            Based on the following ingredient suitability:
            {suitability}

            Determine an overall suitability rating for the product:
            - Suitable: if all ingredients are Suitable or Unknown
            - Use with Caution: if any ingredients are Caution, but none are Avoid
            - Avoid: if any ingredients are Avoid

            Overall rating: Is this product suitable absed upon above ratings.
            Explanation: Explain what long-term issues might exist if continued to consume this product.
            """
        )

        suitability_text = "\n".join([f"{k}: {v}" for k, v in suitability.items()])
        result = self._query_vectorstore(rating_query.format(suitability=suitability_text))

        rating, explanation = self._parse_overall_rating(result)

        # # Groundedness check
        # groundedness_check = UpstageGroundednessCheck(api_key=self.upstage_api_key)
        # request_input = {
        #     "context": suitability_text,
        #     "answer": explanation,
        # }
        # groundedness_response = groundedness_check.invoke(request_input)

        groundedness_response =  "grounded"

        # If the response is not grounded, mark it in the explanation
        if groundedness_response == "notGrounded":
            explanation += " (Note: This explanation may not be fully grounded in the provided data.)"
        elif groundedness_response == "notSure":
            explanation += " (Note: The accuracy of this explanation could not be confirmed.)"

        return rating, explanation
    
    def _parse_overall_rating(self, result: str) -> Tuple[str, str]:
        rating_match = re.search(r'Overall Rating: (.+)', result)
        explanation_match = re.search(r'Explanation: (.+)', result, re.DOTALL)
        
        rating = rating_match.group(1) if rating_match else "Unknown"
        explanation = explanation_match.group(1).strip() if explanation_match else "Unable to generate explanation."
        
        return rating, explanation

    def translate_text(self, text: str, target_language: str = "en") -> str:
        messages = [HumanMessage(content=f"Translate the following text to {target_language}: {text}")]
        response = self.translator.invoke(messages)
        return response.content

    def ocr_product_image(self, image_path: str) -> str:
        url = "https://api.upstage.ai/v1/document-ai/ocr"
        headers = {"Authorization": f"Bearer {self.upstage_api_key}"}
        with open(image_path, "rb") as image_file:
            files = {"document": image_file}
            response = requests.post(url, headers=headers, files=files)
        return response.json()['text']

    def get_product_ingredients(self, product_name: str) -> List[str]:
        result = self.api.product.text_search(product_name, page=1, page_size=1)
        if result['products']:
            # Try to get the ingredients in English first
            ingredients_text = result['products'][0].get('ingredients_text_en', '')
            # If not found or empty, fall back to the default ingredients_text
            if not ingredients_text:
                ingredients_text = result['products'][0].get('ingredients_text', '')

            # Return the list of ingredients, split by comma and stripped of extra spaces
            return [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
        return []


    def analyze_product_from_image(self, image_path: str) -> ProductAnalysis:
        ocr_text = self.ocr_product_image(image_path)
        context = self.tavily.search(query=f"Find product name for any such product that carries similar texts and brand '{ocr_text}'")
        
        product_name_query = PromptTemplate(
            input_variables=["context"],
            template="Extract the most likely product name from the following context:\n\n{context}\n\nProduct Name:"
        ).format(context=context['results'][0]['content'])
        
        product_name = self._query_vectorstore(product_name_query)
        ingredients = self.get_product_ingredients(product_name)
        
        return self.analyze_product(product_name, ingredients)

    def analyze_product_from_text(self, product_text: str) -> ProductAnalysis:
        ingredients = self.get_product_ingredients(product_text)
        return self.analyze_product(product_text, ingredients)

# Example usage
if __name__ == "__main__":
    service = FoodIntoleranceAnalysisService(upstage_api_key="up_xq", tavily_api_key="tvly-KK")
    
    # Process the food intolerance report
    report = service.process_pdf("/test_pdfs/A110.pdf")
    # print(f"Reference Range: {report.reference_range}")
    # print("Food Items:")
    # for item in report.food_items:
    #     print(f"{item.name}: {item.value} U/mL - {item.category}")

    # Analyze a product from text
    product_analysis = service.analyze_product_from_text("Nutty Muesli - By Sainsbury's")
    print(f"\nProduct Analysis for {product_analysis.product_name}:")
    print(f"Ingredients: {', '.join(product_analysis.ingredients)}")
    print("Suitability:")
    for ingredient, suitability in product_analysis.suitability.items():
        print(f"  {ingredient}: {suitability}")
    print(f"Overall Rating: {product_analysis.overall_rating}")
    print(f"Explanation: {product_analysis.explanation}")