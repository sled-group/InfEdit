def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False