import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { trigger, style, animate, transition } from '@angular/animations';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
  animations: [
    trigger('scrollAnimation', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-20px)' }),
        animate('300ms', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
  ],
})
export class HomeComponent {
  showCompage = false;
  response:any;
  rawResponse:any;
  changeList:any = [];
  changeListFinalData:any = [];
  constructor(
    private http: HttpClient,
  ){
    this.getData();
  }

  getData(){
    this.http.get('assets/response.json').subscribe((res:any)=>{
      this.rawResponse = JSON.parse(JSON.stringify(res));
      console.log(res.comparison_output.content);
      this.changeList = this.splitChangeContent(res.comparison_output.content);
    //  console.log(this.changeList);
      console.log(this.changeList);
      this.response = res;
     });
  }

  formateResponse(val:any){
    if(val){
      return val.replace(/\n/g, '<br>')
    } else {
      return "";
    }
     
  }

  splitChangeContent(res:any){
    let splitParentData = res.split("\n\n");
    if(splitParentData.length){
      splitParentData.forEach((element:any) => {
      //  this.splitSubContent(element)
      if(this.splitSubContent(element)){
        this.changeListFinalData.push(this.splitSubContent(element));
      }
        
      });
    }
    return this.changeListFinalData;
  }

  splitSubContent(val:any){
    let data = val.split("\n");
    let retunObj:any = {};
    data.forEach((ele:any) =>{
       let objVal = ele.replace(/['"]+/g, '').split(":");
       console.log(objVal[0]);
       let key = objVal[0];
       let value = objVal[1];
       if(key && value){
        if(key=='Line Content'){
          retunObj.line_content = this.removeLeadingWhitespace(value);
        } else {
          retunObj[key] = value;
        }
       
       }
       
      //  if(objVal.length){
      //   retunObj[objVal[0]] = objVal[1].toString();
      //  }
       //
    })
  //  console.log(data.replace(/['"]+/g, ''));
  console.log(retunObj);
  return retunObj;
  }


   highlightText(text:any) {
    // Get the input string and the target text
   let rawContent = this.rawResponse.pdf_text;

    // Check if the input string exists in the target text
    if (rawContent.includes(text)) {
      // Replace the target text with the highlighted version
      const highlightedText = rawContent.replace(
        new RegExp(text, 'g'),
        `<span class="highlight" id="highlightElement" [@scrollAnimation]>${text}</span>`

         
      );
      
      this.response.pdf_text = this.formateResponse(highlightedText);
      console.log(this.response.pdf_text);
      this.scrollToElement();

    } else {
     console.log('hrer');
    }
  }


  removeLeadingWhitespace(inputString:any) {
    return inputString.replace(/^\s+/, '');
  }

  scrollToElement(): void {
    const element = document.querySelector(".highlight");
    
    console.log(element);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  }

}
