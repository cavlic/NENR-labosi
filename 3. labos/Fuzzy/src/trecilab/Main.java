package trecilab;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.*;
import java.util.Scanner;


public class Main {

    
    public static void main(String[] args) throws IOException {
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
  
        IDefuzzify defuzzifier = new COADefuzzifier();
        FuzzySystem kormilo = new KormiloFuzzySystemMin(defuzzifier);
        FuzzySystem akceleracija = new AkcelFuzzySystemMin(defuzzifier);
        
        kormilo.initRules();
        akceleracija.initRules();
        
       	        
	    int L = 0, D = 0, LK = 0, DK = 0, V = 0, S = 0, a, k;
	    String line;
		while(true){
			if ((line = input.readLine()) != null) {
                if (line.charAt(0) == 'K') break;
                Scanner s = new Scanner(line);
                L = s.nextInt();
                D = s.nextInt();
                LK = s.nextInt();
                DK = s.nextInt();
                V = s.nextInt();
                S = s.nextInt();
                s.close();
                
    	        String output = String.format("%d %d %d %d %d %d", L, D, LK, DK, V, S);
                System.err.println(output);


	        }
			
			a = akceleracija.conclude(L, D, LK, DK, V, S);
			k = kormilo.conclude(L, D, LK, DK, V, S);				
	        System.out.println(a + " " + k);
	        System.out.flush();
	        

	        
	   }
    }

}

