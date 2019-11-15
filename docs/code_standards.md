This section specifies the code standards for the development of new functionalities in the dnn_opt library.

## Variables, identifiers and indentation

  1.1. Use lower camel case all the time, even for class names.
  
  1.2. Variable names and identifiers should be written in English. Avoid the use of contractions.
  
  1.3. Private and protected members should start with underscore `_`.
  
  1.4. Each line should contain no more than 80 characters.
  
  1.5. Use two space indentation, never use tabs.
  
  1.6. Do not indent private, protected, public keywords. Do not indent nested namespaces or classes within.
  
  1.7. For initialization in the constructors, start the initialization list in a new line, including the two dots (:). Prefer one line for each initialized member if you need more than one line.
  

## Blocks

  2.1. Put braces in its own lines.
  
  2.2. Close each namespace with a comment indication: ` // namespace namespace_name `
  
  2.3. Code segments that are designed for local usage should be placed within a void namespace.
  
  2.4. Declare variables at the beginning of its code block, leave an empty line after.
  
  2.5. Separate logical code blocks by empty lines if you feel this will improve readability.
  
  2.6. Leave an empty line before the closing brace of a class and before the return statement of a method implementation.
  

## Comments and documentation

  3.1. Prefer using the enclosed form of comment ` /* this is a comment*/ ` instead of other forms.
  
  3.2. If you need to put some explanatory comment put it in the line above. Leave an empty line if the comment apply for more than one line below. Use this together with 2.5.
  
  3.3. Document all classes and function you write, use the Java `@command` form instead of the `\command`.
  
  3.4. For each class you should specify a @brief, @author, @version and @date. You can optionally put some extended description after the brief in several new paragraphs.
  
  3.5. For each function you should specify a @brief, @param, @throws, @return. You can optionally put some extended description after the brief in several new paragraphs.
  
  3.6. For each member attribute you should declare its purpose above.
  
  3.7. Begin each header file with a copyright disclaimer as a comment.
  
  3.8. Make your best to include ISO 690 references for established algorithms in the extended description of the documentation when applicable.
  

## General guidelines

  4.1. Prefer putting the pointer modifier `*` next to the type declaration.
  
  4.2. Avoid the use of the auto keyword.
  
  4.3. Use a space to separate operators and operands. Do not use space to separate parenthesis.
  
  4.4. Always specify virtual modifier in derived classes.
  
  4.5. Include headers that do not belong to the library first and then include the headers that belong to the library.
  
  4.6. In .cpp files implement the most important methods of the class first, then implement trivial methods (e.g. getters and setters) and finally implement the constructor and destructor of the class.
