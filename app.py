
import streamlit as st
import pandas as pd

from model import train_model


st.title('Machine Learning Classification App')


file = st.file_uploader(
    'Upload your dataset',
    type=['csv']
)

if file is not None:

    df = pd.read_csv(file)

    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    st.subheader('Target Selection')

    target = st.selectbox(
        'Select the target variable',
        df.columns.to_list()
    )

    available_features = [
        col for col in df.columns if col != target
    ]

    st.subheader('Model Selection')

    model_type = st.selectbox(
        'Select the classification model',
        ['KNN', 'Logistic Regression', 'Random Forest']
    )

    st.subheader('Feature Selection')

    feature_mode = st.selectbox(
        'Select feature mode:',
        ['Manual Selection', 'PCA Reduction']
    )

    features = st.multiselect(
        'Select the features',
        available_features
    )

    if len(features) >= 2:

        if feature_mode == 'PCA Reduction':

            max_components = min(
                10,
                len(features),
                int(len(df) * 0.8)
            )

            n_components = st.number_input(
                'Number of PCA components',
                min_value=1,
                max_value=max(1, max_components),
                value=2,
                step=1
            )

            if n_components < 2:

                st.warning(
                    'PCA requires at least 2 components. '
                    'Please select 2 or more components.'
                )

        else:
            n_components = 2

        if model_type == 'KNN':

            n_neighbors = st.slider(
                'Number of Neighbors (K)',
                min_value=1,
                max_value=15,
                value=5
            )

        else:
            n_neighbors = 5

        if n_components >= 2:

            if st.button('Train Model'):

                try:

                    results = train_model(
                        df=df,
                        features=features,
                        target=target,
                        model_type=model_type,
                        feature_mode=feature_mode,
                        n_components=n_components,
                        n_neighbors=n_neighbors
                    )

                    st.subheader('Model Evaluation')

                    st.write(
                        'Accuracy:',
                        results['Accuracy']
                    )

                    st.write(
                        'Precision:',
                        results['Precision']
                    )

                    st.write(
                        'Recall:',
                        results['Recall']
                    )

                    if feature_mode == 'PCA Reduction':

                        st.subheader('PCA Results')

                        st.write(
                            'Explained Variance:',
                            results['Explained Variance']
                        )

                    st.subheader('Prediction Results')

                    comparison = pd.DataFrame({
                        'Actual': results['Actual'].values,
                        'Predicted': results['Predictions']
                    })

                    st.dataframe(comparison)

                    st.success(
                        'Model trained successfully!'
                    )

                except Exception as e:

                    st.error(
                        'Error training model: ' + str(e)
                    )

    else:

        st.warning('Select at least 2 features')

else:

    st.info('Please upload a CSV dataset')
