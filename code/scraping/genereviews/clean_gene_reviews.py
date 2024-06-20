import sys
import lxml.etree as ET

IGNORED_TAGS = set([
    'fig',
    'table-wrap',
    'label',
    'ref-list',
])

IGNORED_SECTIONS = set([
    'Author_Notes',
    'Author_History',
    'Acknowledgements',
    'Revision_History',
    'References',
    'Chapter_Notes',
])

def element_to_str(element):
    return ''.join(element.itertext()).strip()

def traverse_element(element, heading_depth=1, list_depth=0):
    if element.get("specific-use") == "from-external-xml":
        return
    elif element.tag == 'sec' and element.get('id', '').split('.')[-1] in IGNORED_SECTIONS:
        return
    elif element.tag in IGNORED_TAGS:
        return

    if element.tag == 'sec':
        heading_depth += 1
        print('')
    elif element.tag == 'title':
        print(f"{'#' * heading_depth} {element_to_str(element)}")
        return
    elif element.tag == 'list':
        list_depth += 1
    elif element.tag == 'list-item':
        print(f"{'  ' * list_depth}- ", end="")
    elif element.tag == 'p':
        print(f"{element_to_str(element)}")
        return

    for child in element:
        traverse_element(child, heading_depth=heading_depth, list_depth=list_depth)

def convert_xml_to_text(xml_file, xpaths):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for xpath in xpaths:
        elements = root.xpath(xpath)
        for element in elements:
            traverse_element(element)

if __name__ == "__main__":
    xml_file = sys.argv[1]
    xpaths = [
        "/book-part-wrapper/book-part/book-part-meta/abstract",
        "/book-part-wrapper/book-part/body",
    ]
    convert_xml_to_text(xml_file, xpaths)
