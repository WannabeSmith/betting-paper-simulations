(TeX-add-style-hook
 "notes"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("geometry" "margin=0.85in") ("natbib" "numbers") ("hyperref" "colorlinks" "linkcolor=magenta" "citecolor=blue" "pagebackref=true") ("enumitem" "shortlabels") ("fontenc" "T1")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "geometry"
    "natbib"
    "amsmath"
    "amsthm"
    "amsfonts"
    "hyperref"
    "graphicx"
    "enumitem"
    "fontenc")
   (TeX-add-symbols
    "EE"
    "PP"
    "QQ"
    "Acal"
    "Xcal"
    "wor"))
 :latex)

