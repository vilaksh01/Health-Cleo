import re
import streamlit as st
import requests
import csv
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_upstage import ChatUpstage, UpstageGroundednessCheck
import pandas as pd
from tavily import TavilyClient
import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import plotly.graph_objects as go

# def visualize_graph(G: nx.DiGraph, title: str) -> plt.Figure:
#     # Use graphviz_layout for a hierarchical layout
#     pos = graphviz_layout(G, prog='dot')
    
#     # Create a larger figure
#     fig, ax = plt.subplots(figsize=(20, 16))
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8, ax=ax)
#     nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
    
#     # Add edge labels
#     edge_labels = nx.get_edge_attributes(G, 'description')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    
#     # Remove axis
#     ax.axis('off')
    
#     # Add title
#     plt.title(title, fontsize=16, fontweight='bold')
    
#     # Adjust layout
#     plt.tight_layout()
    
#     return fig

import plotly.graph_objects as go
import networkx as nx

def visualize_graph_interactive(G: nx.DiGraph, title: str) -> go.Figure:
    # Use spring layout for node positioning
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node traces
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[],
        textposition="top center"
    )
    
    # Color node points by the number of connections
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f"{adjacencies[0]}<br># of connections: {len(adjacencies[1])}")
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Add edge labels
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_annotation(
            x=(x0+x1)/2,
            y=(y0+y1)/2,
            text=edge[2].get('description', ''),
            showarrow=False,
            font=dict(size=8),
            bgcolor="white",
            opacity=0.8
        )
    
    return fig

def truncate_label(label: str, max_length: int = 20) -> str:
    """Truncate long labels to improve readability."""
    return label if len(label) <= max_length else label[:max_length-3] + '...'

def process_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Process the graph to improve visualization."""
    # Truncate long node labels
    mapping = {node: truncate_label(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    
    # Truncate long edge descriptions
    for u, v, data in G.edges(data=True):
        data['description'] = truncate_label(data.get('description', ''), 30)
    
    return G

# Define data models
class Nutrient(BaseModel):
    id: str
    name: str
    unit_name: str
    nutrient_nbr: str
    rank: str

class MappedNutrient(BaseModel):
    id: str
    name: str
    unit_name: str
    value: float
    nutrient_nbr: str

class NutritionInfo(BaseModel):
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None
    fiber: Optional[float] = None
    sugar: Optional[float] = None
    mapped_nutrients: List[MappedNutrient]

class FoodItem(BaseModel):
    name: str
    serving_qty: float
    serving_unit: str
    serving_weight_grams: float
    nutrition_info: NutritionInfo

class FoodIntoleranceItem(BaseModel):
    name: str
    value: float
    category: str

class AllergyItem(BaseModel):
    name: str
    value: float
    category: str

class AllergyReport(BaseModel):
    total_ige: float
    allergy_items: List[AllergyItem]

class EnhancedFoodIntoleranceReport(BaseModel):
    food_items: List[FoodIntoleranceItem]
    allergy_report: AllergyReport

class UserProfile(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    activity_level: str
    dietary_restrictions: List[str]

class EnhancedFoodAnalysis:
    def __init__(self, upstage_api_key: str, nutritionix_app_id: str, nutritionix_api_key: str, tavily_api_key: str):
        self.upstage_api_key = upstage_api_key
        self.nutritionix_app_id = nutritionix_app_id
        self.nutritionix_api_key = nutritionix_api_key
        self.llm = ChatUpstage(api_key=upstage_api_key)
        self.groundedness_check = UpstageGroundednessCheck(api_key=upstage_api_key)
        self.enhanced_food_intolerance_report: Optional[EnhancedFoodIntoleranceReport] = None
        self.nutrients_dict = self.load_nutrients()
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.user_profile: Optional[UserProfile] = None

    def set_enhanced_food_intolerance_report(self, report: EnhancedFoodIntoleranceReport):
        self.enhanced_food_intolerance_report = report

    def set_user_profile(self, profile: UserProfile):
        self.user_profile = profile

    def load_nutrients(self) -> Dict[str, Nutrient]:
        nutrients = {}
        with open('nutrient.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                nutrient = Nutrient(**row)
                nutrients[nutrient.id] = nutrient
                nutrients[nutrient.nutrient_nbr] = nutrient
        return nutrients

    def map_full_nutrients(self, full_nutrients: List[Dict[str, float]]) -> List[MappedNutrient]:
        mapped_nutrients = []
        for nutrient in full_nutrients:
            attr_id = str(nutrient['attr_id'])
            if attr_id in self.nutrients_dict:
                nutrient_info = self.nutrients_dict[attr_id]
                mapped_nutrient = MappedNutrient(
                    id=attr_id,
                    name=nutrient_info.name,
                    unit_name=nutrient_info.unit_name,
                    value=nutrient['value'],
                    nutrient_nbr=nutrient_info.nutrient_nbr
                )
                mapped_nutrients.append(mapped_nutrient)
        return mapped_nutrients

    def get_nutrition_info(self, query: str) -> List[FoodItem]:
        url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
        headers = {
            "x-app-id": self.nutritionix_app_id,
            "x-app-key": self.nutritionix_api_key,
            "Content-Type": "application/json"
        }
        data = {"query": query}
        
        response = requests.post(url, headers=headers, json=data)
        food_items = []
        if response.status_code == 200:
            result = response.json()
            if 'foods' in result:
                for food in result['foods']:
                    mapped_nutrients = self.map_full_nutrients(food['full_nutrients'])
                    nutrition_info = NutritionInfo(
                        calories=food.get('nf_calories'),
                        protein=food.get('nf_protein'),
                        carbs=food.get('nf_total_carbohydrate'),
                        fat=food.get('nf_total_fat'),
                        fiber=food.get('nf_dietary_fiber'),
                        sugar=food.get('nf_sugars'),
                        mapped_nutrients=mapped_nutrients
                    )
                    food_item = FoodItem(
                        name=food['food_name'],
                        serving_qty=food['serving_qty'],
                        serving_unit=food['serving_unit'],
                        serving_weight_grams=food['serving_weight_grams'],
                        nutrition_info=nutrition_info
                    )
                    food_items.append(food_item)
        return food_items

    def combine_nutrition_info(self, food_items: List[FoodItem]) -> NutritionInfo:
        combined = NutritionInfo(
            calories=0, protein=0, carbs=0, fat=0, fiber=0, sugar=0, mapped_nutrients=[]
        )
        nutrient_dict = {}

        for item in food_items:
            for attr in ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']:
                value = getattr(item.nutrition_info, attr)
                if value is not None:
                    current = getattr(combined, attr) or 0
                    setattr(combined, attr, current + value)

            for nutrient in item.nutrition_info.mapped_nutrients:
                if nutrient.id in nutrient_dict:
                    nutrient_dict[nutrient.id].value += nutrient.value
                else:
                    nutrient_dict[nutrient.id] = MappedNutrient(
                        id=nutrient.id,
                        name=nutrient.name,
                        unit_name=nutrient.unit_name,
                        value=nutrient.value,
                        nutrient_nbr=nutrient.nutrient_nbr
                    )

        combined.mapped_nutrients = [n for n in nutrient_dict.values() if n.value > 0]
        return combined

    def get_population_context(self, food_query: str) -> List[str]:
        search_query = f"Long-term effects of {food_query} consumption on population health, including regional impacts, adverse symptoms, and potential health benefits over time"

        if self.user_profile:
            search_query += f" on {self.user_profile.age} year old {self.user_profile.gender}"
        search_results = self.tavily_client.search(query=search_query)
        st.text(search_results)
        return [result['content'] for result in search_results['results'][:3]]
    
    def analyze_food(self, food_query: str) -> Dict[str, any]:
        food_items = self.get_nutrition_info(food_query)
        combined_nutrition = self.combine_nutrition_info(food_items)
        population_context = self.get_population_context(food_query)
        
        explanation = self._generate_explanation(food_query, food_items, combined_nutrition, population_context)
        recommendations = self._generate_recommendations(food_query, food_items, population_context)
        
        intolerant_items = []
        allergic_items = []
        if self.enhanced_food_intolerance_report:
            intolerant_items = [item.name for item in self.enhanced_food_intolerance_report.food_items if item.category in ["Elevated", "Borderline"]]
            allergic_items = [item.name for item in self.enhanced_food_intolerance_report.allergy_report.allergy_items if item.category in ["Elevated", "Borderline"]]
        
        # causal_graph = self.generate_causal_graph(food_query, intolerant_items, allergic_items, food_items, population_context)
        # inference_graph = self.generate_inference_graph(food_query, intolerant_items, allergic_items, food_items, population_context)
        
        causal_graph = self.generate_causal_graph(food_query, intolerant_items, allergic_items, food_items, population_context)
        inference_graph = self.generate_inference_graph(food_query, intolerant_items, allergic_items, food_items, population_context)
    
        # Process graphs for better visualization
        causal_graph_processed = process_graph(causal_graph)
        inference_graph_processed = process_graph(inference_graph)
        
        return {
            "food_query": food_query,
            "food_items": food_items,
            "combined_nutrition": combined_nutrition,
            "population_context": population_context,
            "explanation": explanation,
            "recommendations": recommendations,
            "causal_graph": causal_graph_processed,
            "inference_graph": inference_graph_processed
        }

    def _generate_explanation(self, food_query: str, food_items: List[FoodItem], combined_nutrition: NutritionInfo, population_context: str) -> str:
        food_items_str = "\n".join([f"- {item.name}: {item.serving_qty} {item.serving_unit} ({item.serving_weight_grams}g)" for item in food_items])
        mapped_nutrients_str = "\n".join([f"{n.name} (Nutrient #{n.nutrient_nbr}): {n.value:.2f} {n.unit_name}" for n in combined_nutrition.mapped_nutrients if n.value > 0])
        
        user_profile_str = ""
        if self.user_profile:
            user_profile_str = f"""
            User Profile:
            Age: {self.user_profile.age}
            Gender: {self.user_profile.gender}
            Height: {self.user_profile.height} cm
            Weight: {self.user_profile.weight} kg
            Activity Level: {self.user_profile.activity_level}
            Dietary Restrictions: {', '.join(self.user_profile.dietary_restrictions)}
            """

        combined_nutrition_str = "\n".join([
            f"{attr.capitalize()}: {getattr(combined_nutrition, attr):.2f} {'g' if attr != 'calories' else ''}" 
            if getattr(combined_nutrition, attr) is not None else f"{attr.capitalize()}: N/A"
            for attr in ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']
        ])

        prompt = f"""
        Analyze the following food items and their combined nutritional information:

        Food Query: {food_query}

        Individual Food Items:
        {food_items_str}

        Combined Nutritional Information:
        {combined_nutrition_str}

        Additional nutritional information (non-zero values only):
        {mapped_nutrients_str}

        {user_profile_str}

        Population Context:
        {population_context}

        Provide a detailed explanation of the nutritional content and its potential impact on health, 
        considering the user's profile (if available) and the population context. Include information about:
        1. Overall nutritional value of the combination
        2. Potential allergens or irritants in any of the food items
        3. Suitability for different dietary needs (e.g., low-carb, high-protein, etc.)
        4. Any precautions or considerations for consumption
        5. Notable nutrients and their health benefits or concerns
        6. How the combination of these foods might affect digestion or nutrient absorption
        7. Relevance to the user's profile (if available) and general population trends
        8. Note any missing nutritional information and its potential implications

        Format your response in markdown for better readability.
        """
        return self.llm.invoke(prompt)

    def _generate_recommendations(self, food_query: str, food_items: List[FoodItem], population_context: str) -> List[str]:
        intolerant_items = []
        allergic_items = []
        if self.enhanced_food_intolerance_report:
            intolerant_items = [item.name for item in self.enhanced_food_intolerance_report.food_items if item.category in ["Elevated", "Borderline"]]
            allergic_items = [item.name for item in self.enhanced_food_intolerance_report.allergy_report.allergy_items if item.category in ["Elevated", "Borderline"]]
        
        food_items_str = ", ".join([item.name for item in food_items])
        
        user_profile_str = ""
        if self.user_profile:
            user_profile_str = f"""
            User Profile:
            Age: {self.user_profile.age}
            Gender: {self.user_profile.gender}
            Height: {self.user_profile.height} cm
            Weight: {self.user_profile.weight} kg
            Activity Level: {self.user_profile.activity_level}
            Dietary Restrictions: {', '.join(self.user_profile.dietary_restrictions)}
            """
        
        prompt = f"""
        Consider the following food items: {food_items_str}

        The user has reported intolerances or sensitivities to the following items:
        {', '.join(intolerant_items)}

        The user has reported allergies to the following items:
        {', '.join(allergic_items)}

        {user_profile_str}

        Population Context:
        {population_context}

        Provide 3-5 recommendations for this user, considering their food intolerances, allergies, profile (if available), and the population context. Include:
        1. Whether each food item is likely safe or should be avoided
        2. Potential substitutes for any items that should be avoided
        3. Tips for preparation or consumption to minimize potential issues
        4. Any other relevant advice for someone with these specific characteristics
        5. How the recommendations compare to general population guidelines

        Format your response as a list of markdown bullet points.
        """
        response = self.llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        recommendations = [rec.strip() for rec in content.split('\n') if rec.strip()]
        
        return recommendations

    def generate_causal_graph(self, food_query: str, intolerant_items: List[str], allergic_items: List[str], food_items: List[FoodItem], population_context: List[str]) -> nx.DiGraph:
        food_items_str = ", ".join([item.name for item in food_items])
        population_context_str = "\n".join(population_context)
        prompt = f"""
        Create a causal graph showing the relationship between consuming {food_items_str} 
        and potential biological effects, considering intolerances to {', '.join(intolerant_items)},
        allergies to {', '.join(allergic_items)}, and the following population context:

        {population_context_str}
        
        The graph should:
        1. Start with the consumption of the food items
        2. Show how they're broken down in the digestive system
        3. Illustrate the interaction with intolerant substances and allergens
        4. Depict the resulting biological effects or symptoms
        5. Include long-term population-level effects based on the provided context
        6. Show any regional impacts or variations mentioned in the population context
        
        Provide the graph as a series of edges in the format:
        node1 -> node2: description
        node2 -> node3: description
        ...
        
        Ensure to include nodes and edges that represent the population-level insights from the context provided.
        """
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        G = nx.DiGraph()
        for line in content.split('\n'):
            if '->' in line:
                source, rest = line.split('->')
                target, description = rest.split(':', 1)
                G.add_edge(source.strip(), target.strip(), description=description.strip())
        
        return G

    def generate_inference_graph(self, food_query: str, intolerant_items: List[str], allergic_items: List[str], food_items: List[FoodItem], population_context: List[str]) -> nx.DiGraph:
        food_items_str = ", ".join([item.name for item in food_items])
        population_context_str = "\n".join(population_context)
        prompt = f"""
        Create an inference graph showing the decision-making process for determining if {food_items_str} 
        are safe to consume, considering intolerances to {', '.join(intolerant_items)},
        allergies to {', '.join(allergic_items)}, and the following population context:

        {population_context_str}
        
        The graph should:
        1. Start with the food items and known intolerances and allergies
        2. Show the decision-making process
        3. Include factors like ingredient analysis and potential cross-contamination
        4. Incorporate population-level insights from the provided context
        5. Consider long-term effects and regional variations mentioned in the context
        6. Conclude with a safety assessment for each food item, both for individuals and at a population level

        Provide the graph as a series of edges in the format:
        node1 -> node2: description
        node2 -> node3: description
        ...
        
        Ensure to include nodes and edges that represent the decision-making process based on both individual factors and population-level insights.
        """
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        G = nx.DiGraph()
        for line in content.split('\n'):
            if '->' in line:
                source, rest = line.split('->')
                target, description = rest.split(':', 1)
                G.add_edge(source.strip(), target.strip(), description=description.strip())
        
        return G

    def parse_enhanced_food_intolerance_report(self, report_text: str, allergy_text: str) -> EnhancedFoodIntoleranceReport:
        prompt = f"""
        Parse the following food intolerance report and allergy report and extract the relevant information, be precise don't add any extra items which is not present:

        Food Intolerance Report:
        {report_text}

        Allergy Report:
        {allergy_text}

        For each food item mentioned in the food intolerance report, provide:
        1. The name of the food item
        2. Its intolerance value (if available)
        3. Its category (Elevated, Borderline, or Normal)

        For the allergy report, provide:
        1. The total IgE value
        2. For each allergen mentioned, provide:
           a. The name of the allergen
           b. Its value
           c. Its category (Elevated, Borderline, or Normal)
  
        Format your response as a JSON object with the following structure:
        {{
            "food_items": [
                {{"name": "Food Item 1", "value": value, "category": "Elevated"}},
                {{"name": "Food Item 2", "value": value, "category": "Borderline"}},
                ...
            ],
            "allergy_report": {{
                "total_ige": value ,
                "allergy_items": [
                    {{"name": "Allergen 1", "value": value, "category": "Elevated"}},
                    {{"name": "Allergen 2", "value": value, "category": "Normal"}},
                    ...
                ]
            }}
        }}
        """
        response = self.llm.invoke(prompt)
        
        content = response.content if hasattr(response, 'content') else str(response)

        try:
            parsed_data = json.loads(content)
            return EnhancedFoodIntoleranceReport(**parsed_data)
        except json.JSONDecodeError:
            st.error("Failed to parse the food intolerance and allergy report. Please check the format and try again.")
            return EnhancedFoodIntoleranceReport(food_items=[], allergy_report=AllergyReport(total_ige=0, allergy_items=[]))

# Streamlit app
def main():
    st.title("Enhanced Food Intolerance and Allergy Analysis")

    # Sidebar for API keys
    st.sidebar.header("API Configuration")
    upstage_api_key = st.sidebar.text_input("Upstage API Key", value="up_1oFggobss8MgZyAAVXDYNNJ4fWdvC", type="password")
    nutritionix_app_id = st.sidebar.text_input("Nutritionix App ID", value="faee4d9c", type="password")
    nutritionix_api_key = st.sidebar.text_input("Nutritionix API Key", value="fa0407063db1f2dc54d4f751ef894665", type="password")
    tavily_api_key = st.sidebar.text_input("Tavily API Key", value="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", type="password")

    # Initialize the analyzer
    if upstage_api_key and nutritionix_app_id and nutritionix_api_key and tavily_api_key:
        analyzer = EnhancedFoodAnalysis(upstage_api_key, nutritionix_app_id, nutritionix_api_key, tavily_api_key)
    else:
        st.warning("Please enter all API keys to proceed.")
        return

    # User Profile Input
    st.header("User Profile")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=500.0)
    activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"])
    dietary_restrictions = st.multiselect("Dietary Restrictions", ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut-Free", "Low-Carb", "Low-Fat"])

    user_profile = UserProfile(
        age=age,
        gender=gender,
        height=height,
        weight=weight,
        activity_level=activity_level,
        dietary_restrictions=dietary_restrictions
    )
    analyzer.set_user_profile(user_profile)

    # Food Intolerance and Allergy Report
    st.header("Food Intolerance and Allergy Report")
    report_input_method = st.radio("Choose input method:", ["Manual Entry", "Parse Report"])

    if report_input_method == "Manual Entry":
        st.write("Enter food intolerances (one per line, format: Name,Value,Category)")
        st.write("Example: Gluten,0.8,Elevated")
        intolerance_items = st.text_area("Food Intolerances")
        
        st.write("Enter allergies (one per line, format: Name,Value,Category)")
        st.write("Example: Peanuts,0.9,Elevated")
        allergy_items = st.text_area("Allergies")
        
        total_ige = st.number_input("Total IgE", min_value=0.0, value=0.0)
        
        if intolerance_items or allergy_items:
            food_items = []
            allergy_list = []
            
            if intolerance_items:
                for line in intolerance_items.split('\n'):
                    if line.strip():  # Check if the line is not empty
                        try:
                            name, value, category = line.split(',')
                            food_items.append(FoodIntoleranceItem(
                                name=name.strip(),
                                value=float(value),
                                category=category.strip()
                            ))
                        except ValueError:
                            st.error(f"Invalid format for intolerance item: {line}. Please use 'Name,Value,Category'.")
            
            if allergy_items:
                for line in allergy_items.split('\n'):
                    if line.strip():  # Check if the line is not empty
                        try:
                            name, value, category = line.split(',')
                            allergy_list.append(AllergyItem(
                                name=name.strip(),
                                value=float(value),
                                category=category.strip()
                            ))
                        except ValueError:
                            st.error(f"Invalid format for allergy item: {line}. Please use 'Name,Value,Category'.")
            
            if food_items or allergy_list:
                allergy_report = AllergyReport(total_ige=total_ige, allergy_items=allergy_list)
                analyzer.set_enhanced_food_intolerance_report(
                    EnhancedFoodIntoleranceReport(food_items=food_items, allergy_report=allergy_report)
                )
                st.success("Food intolerance and allergy information updated successfully.")
    
    else:
        report_text = st.text_area("Paste your food intolerance report here:")
        allergy_text = st.text_area("Paste your allergy report here:")
        if report_text and allergy_text:
            parsed_report = analyzer.parse_enhanced_food_intolerance_report(report_text, allergy_text)
            analyzer.set_enhanced_food_intolerance_report(parsed_report)
            st.write("Parsed Food Intolerance and Allergy Report:")
            st.write(parsed_report)

    # Food Query Analysis
    st.header("Food Analysis")
    food_query = st.text_input("Enter food items to analyze")
    if st.button("Analyze") and food_query:
        with st.spinner("Analyzing food..."):
            result = analyzer.analyze_food(food_query)

        st.subheader("Food Items")
        for item in result['food_items']:
            st.write(f"- {item.name}: {item.serving_qty} {item.serving_unit} ({item.serving_weight_grams}g)")

        st.subheader("Combined Nutritional Information")
        combined = result['combined_nutrition']

        main_nutrients = pd.DataFrame({
            'Nutrient': ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar'],
            'Value': [
                f"{combined.calories:.2f}" if combined.calories is not None else "N/A",
                f"{combined.protein:.2f}g" if combined.protein is not None else "N/A",
                f"{combined.carbs:.2f}g" if combined.carbs is not None else "N/A",
                f"{combined.fat:.2f}g" if combined.fat is not None else "N/A",
                f"{combined.fiber:.2f}g" if combined.fiber is not None else "N/A",
                f"{combined.sugar:.2f}g" if combined.sugar is not None else "N/A"
            ]
        })
        st.table(main_nutrients)

        st.subheader("Detailed Nutritional Breakdown")
        detailed_nutrients = pd.DataFrame([
            {
                'Nutrient': nutrient.name,
                'Value': f"{nutrient.value:.2f}",
                'Unit': nutrient.unit_name,
                'Nutrient #': nutrient.nutrient_nbr
            }
            for nutrient in combined.mapped_nutrients if nutrient.value > 0
        ])
        st.dataframe(detailed_nutrients, hide_index=True)

        st.subheader("Population Context")
        st.write(result['population_context'])

        st.subheader("Explanation")
        explanation = result['explanation']
        if isinstance(explanation, str):
            formatted_explanation = explanation
        else:
            formatted_explanation = explanation.content if hasattr(explanation, 'content') else str(explanation)
        
        formatted_explanation = formatted_explanation.replace('\\n', '\n').strip('"')
        st.markdown(formatted_explanation)

        st.subheader("Recommendations")
        for rec in result['recommendations']:
            st.markdown(f"- {rec}")

        st.subheader("Causal Graph (Including Population Context)")
        causal_fig = visualize_graph_interactive(result['causal_graph'], "Causal Graph: Food Impact and Population Effects")
        st.plotly_chart(causal_fig, use_container_width=True)

        st.subheader("Inference Graph (Including Population Context)")
        inference_fig = visualize_graph_interactive(result['inference_graph'], "Inference Graph: Food Safety Decision Process")
        st.plotly_chart(inference_fig, use_container_width=True)

if __name__ == "__main__":
    main()


