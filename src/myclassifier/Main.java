/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.io.File;
import weka.core.Instances;

/**
 *
 * @author toshiba
 */
public class Main {
    
    public static void main(String[] args) throws Exception {
        System.out.println("C");
        File f = new File("weather.nominal.arrf");
        if(f.exists() && !f.isDirectory()) { 
            System.out.println("A");
        } else {
            System.out.println("B");
        }
        System.out.println("C");
        
        WekaAccessor access = new WekaAccessor();
        Instances train_data = access.loadArff("weather.nominal.arrf");
        train_data.toString();
        train_data.firstInstance().toString();
        MyId3 id3 = new MyId3();
        id3.buildClassifier(train_data);
        access.tenFoldCrossValidation(id3, train_data);
    }
    
}
