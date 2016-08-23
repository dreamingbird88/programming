/** TODO: http://go/java-style#javadoc

# java MyFirstJava This is my first Java. Yeah!

javac ~/MyFirstJava.java
java MyFirstJava whatareyoudoing?

**/

import java.util.Scanner;
import java.util.Random;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class MyFirstJava {
  public static void main(String[] args) {
    // Note that char/int is not an object class. So
    //     Map<char, int> m = new HashMap<char, int>();
    // is wrong.
    Map<Character, Integer> m = new HashMap<Character, Integer>();
    char[] s = args[0].toCharArray(); // String to charArray.
    for (char c: s) {
      if (m.containsKey(c)) {
        m.put(c, m.get(c) + 1);
      } else {
        m.put(c, 1);
      }
    }
    for (Character c: m.keySet()) {
      if (m.get(c) > 1) {
        System.out.println(c+": "+m.get(c));
      }
    }

    if (true) return;
    // Random number.
    Random r = new Random();
    for (int i = 0; i < 10; ++i) {
      System.out.println(r.nextInt(100));
    }

    // Math lib.
    System.out.println(Math.PI);

    // Input and output.
    System.out.println("Please input your positive integer number: ");
    Scanner in = new Scanner(System.in);
    while (in.hasNextInt()) {
      int input_int = 0;
      int output_int = 0;
      input_int = in.nextInt();
      while (input_int != 0) {
        output_int *= 10;
        output_int += input_int % 10;
        input_int /= 10;
      }
      System.out.println("The reverse number is: " + output_int);
    }

  }

}


