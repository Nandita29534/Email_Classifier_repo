{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a3a74a22-b746-4462-8e5b-2781f0e6dacf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "40256c08-f113-4e69-87d6-6a3b7cb6b62e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-09-03 13:20:22.689 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-03 13:20:22.691 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-03 13:20:27.107 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run D:\\MiniForge\\envs\\email_classification_env\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-09-03 13:20:27.108 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-03 13:20:27.109 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-03 13:20:27.112 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-03 13:20:27.114 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-03 13:20:27.116 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.set_page_config(page_title=\"Email Classifier\", page_icon=\"📧\")\n",
    "\n",
    "st.title(\"📧 Email Classifier Dashboard\")\n",
    "st.write(\"Welcome! Use the sidebar to navigate between pages.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f43f31b6-4592-4cdc-b900-531aad514829",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (email_classification_env)",
   "language": "python",
   "name": "email_classification_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
