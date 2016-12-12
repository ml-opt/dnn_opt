 
for f in ./xml/*.xml;
  do pandoc "$f" -s -f markdown_github -o "${f%}.md";
done 