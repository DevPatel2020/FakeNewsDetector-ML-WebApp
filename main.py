# Module 1: Import necessary packages
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
import warnings
import streamlit_lottie
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

warnings.filterwarnings("ignore")

# Module 2: Load and split the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("fake_or_real_news.csv")
    data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

# Module 3: Select Vectorizer and Classifier
def select_model():
    vectorizer_type = st.sidebar.selectbox("Select Vectorizer", 
        ["TF-IDF", "Bag of Words", "Hashing Vectorizer", "TF-IDF Bigram", "TF-IDF Trigram"])
    classifier_type = st.sidebar.selectbox("Select Classifier", 
        ["Linear SVM", "Naive Bayes", "Logistic Regression", "Random Forest", "Passive Aggressive"])
    return vectorizer_type, classifier_type

# Module 4: Train the model
@st.cache_resource
def train_model(train_data, vectorizer_type, classifier_type):
    # Create vectorizer
    if vectorizer_type == "TF-IDF":
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    elif vectorizer_type == "Bag of Words":
        vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
    elif vectorizer_type == "Hashing Vectorizer":
        vectorizer = HashingVectorizer(stop_words='english', n_features=2**18)
    elif vectorizer_type == "TF-IDF Bigram":
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.7)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_df=0.7)
    
    # Create classifier
    if classifier_type == "Linear SVM":
        classifier = LinearSVC()
    elif classifier_type == "Naive Bayes":
        classifier = MultinomialNB()
    elif classifier_type == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000)
    elif classifier_type == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=100)
    else:
        classifier = PassiveAggressiveClassifier(max_iter=1000)
    
    # Train model
    x_train = vectorizer.fit_transform(train_data['text'])
    classifier.fit(x_train, train_data['fake'])
    
    return vectorizer, classifier

# Module 5: Evaluation Metrics
def get_metrics(test_data, vectorizer, classifier):
    x_test = vectorizer.transform(test_data['text'])
    y_true = test_data['fake']
    y_pred = classifier.predict(x_test)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }

# Update Module 6: Enhanced Visualization Dashboard
def show_dashboard(data):
    # Existing visualizations
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='fake', data=data, palette="Set2", ax=ax)
        ax.set_xticklabels(['Real', 'Fake'])
        ax.set_title("Class Distribution")
        st.pyplot(fig)

    with col2:
        # Word Cloud
        st.subheader("Word Cloud")
        text = " ".join(data['text'])
        wordcloud = WordCloud(stopwords='english', max_words=100, 
                            background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())

    # New visualizations
    st.subheader("Text Analysis")
    plot_word_frequency(data)

# New Module: Model Performance Visualizations
def show_model_insights(model, vectorizer, test_data):
    st.subheader("Model Diagnostics")
    
    # Generate predictions and scores
    X_test = vectorizer.transform(test_data['text'])
    y_true = test_data['fake']
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    st.markdown("### Classification Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        plot_confusion_matrix(y_true, y_pred)
    
    with col2:
        plot_pr_curve(model, X_test, y_true)
    
    # Feature Importance
    st.markdown("### Key Predictors")
    plot_feature_importance(vectorizer, model)
    
    # Calibration Curve
    st.markdown("### Model Calibration")
    plot_calibration_curve(model, X_test, y_true)

# New visualization functions
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def plot_pr_curve(model, X_test, y_true):
    # Handle classifiers without predict_proba
    if isinstance(model, (LinearSVC, PassiveAggressiveClassifier)):
        y_scores = model.decision_function(X_test)
    else:
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_scores = model.decision_function(X_test)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, marker='.')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    st.pyplot(fig)

def plot_feature_importance(vectorizer, model, top_n=15):
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        st.warning("Feature names not available for this vectorizer")
        return

    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
    elif hasattr(model, 'feature_log_prob_'):
        importances = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        st.warning("Feature importance not available for this classifier")
        return
    
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
    ax.set_title(f'Top {top_n} Predictive Features')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)

def plot_word_frequency(data, top_n=15):
    cv = CountVectorizer(stop_words='english', max_features=2000)
    cv.fit(data['text'])
    
    real_counts = cv.transform(data[data.fake == 0].text).sum(axis=0).A1
    fake_counts = cv.transform(data[data.fake == 1].text).sum(axis=0).A1
    features = cv.get_feature_names_out()
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Real News Words
    real_idx = np.argsort(real_counts)[-top_n:]
    sns.barplot(x=real_counts[real_idx], y=features[real_idx], ax=ax[0], palette="Blues_d")
    ax[0].set_title('Common Words in Real News')
    
    # Fake News Words
    fake_idx = np.argsort(fake_counts)[-top_n:]
    sns.barplot(x=fake_counts[fake_idx], y=features[fake_idx], ax=ax[1], palette="Reds_d")
    ax[1].set_title('Common Words in Fake News')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_calibration_curve(model, X_test, y_true):
    if hasattr(model, "predict_proba"):
        prob_pos = model.predict_proba(X_test)[:, 1]
    else:
        prob_pos = model.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, prob_pos, n_bins=10)
    
    fig, ax = plt.subplots()
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Value")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    st.pyplot(fig)

# Module 7: Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon=":newspaper:",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load data
    train_data, test_data = load_data()

    # Sidebar controls
    st.sidebar.header("Model Configuration")
    vectorizer_type, classifier_type = select_model()


    # Main interface
    st.title("Fake News Detection :newspaper:")
    st.markdown("""
    ### 🔍 About This App  
This web application helps detect fake news articles using machine learning models.  
Simply paste a news article, and the system will analyze it using advanced NLP techniques, providing insights into whether the news is real or fake.  
You can also explore model performance metrics and visualizations for a deeper understanding.  
""")

    # User input
    user_input = st.text_area("Paste news article here:", height=200)

    # Prediction section - SINGLE button
    if st.button("Analyze Article"):
        with st.spinner("Training model..."):
            vectorizer, model = train_model(train_data, vectorizer_type, classifier_type)
        
        with st.spinner("Analyzing..."):
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]
            
            if prediction == 1:
                st.error("🚩 This news article appears to be fake!")
            else:
                st.success("✅ This news article seems authentic!")

        # Show metrics
        metrics = get_metrics(test_data, vectorizer, model)
        st.sidebar.subheader("Model Performance")
        st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.sidebar.metric("F1 Score", f"{metrics['f1']:.2%}")
        st.sidebar.metric("Recall", f"{metrics['recall']:.2%}")

        # Show enhanced visualizations
        show_model_insights(model, vectorizer, test_data)

    # Visualizations
    show_dashboard(train_data)

    # Footer
    st.markdown("---")
    

if __name__ == "__main__":
    main()

#cd Fake-News-Detection,venv\Scripts\Activate,streamlit run main.py --client.showErrorDetails=false
# to fix lottie animation