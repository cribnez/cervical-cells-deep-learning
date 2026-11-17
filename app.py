"""
Entry point to launch the Clinical Decision Support Graphical User Interface (GUI).
Run with: python app.py
"""
import tkinter as tk
from src.gui import AplicacionClasificador

def main():
    root = tk.Tk()
    app = AplicacionClasificador(root)
    root.mainloop()

if __name__ == "__main__":
    main()
