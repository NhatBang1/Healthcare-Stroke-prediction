import docx
import os

def remove_names_and_university():
    file_path = r"D:\Bang\Learning\FPT\Semester_8\DSP391\PRJ2\Healthcare-Stroke-prediction\Documents\Paper_ICPU.docx"
    doc = docx.Document(file_path)
    
    # We found the names at index 1 and university at index 2
    
    # Find the paragraphs by text to be safe
    elements_to_remove = []
    for p in doc.paragraphs:
        if "Nguyen Nhat Bang" in p.text or "FPT University" in p.text:
            elements_to_remove.append(p._element)
            
    for el in elements_to_remove:
        el.getparent().remove(el)
            
    doc.save(file_path)
    print("Successfully removed names and university from Paper_ICPU.docx")

if __name__ == "__main__":
    remove_names_and_university()
